import random 
import torch
from torch.nn.functional import batch_norm 
from torch.optim import Adam 
from torch.optim.lr_scheduler import LambdaLR 
from torch.nn import NLLLoss 

import argparse, os, pickle 
import numpy as np
from tqdm import tqdm 
from torch.utils.data import dataloader 
from model.transformer import Transformer 
import multiprocessing
import itertools 
from shutil import copyfile 

from data import ImageDetectionsField, TextField, RawField
from data import DataLoader, COCO 
from model import Transformer, VisualEncoder, CaptionDecoder, ScaledDotProductAttention
import evaluation 
from evaluation import PTBTokenizer, Cider
import warnings
warnings.filterwarnings("ignore")


random.seed(2021)
torch.manual_seed(2021)
np.random.seed(2021)


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):
            detections, captions = detections.to(device), captions.to(device) 
            out = model(detections, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step() 
            if it == 9:
                break 
    scheduler.step() 

    loss = running_loss / len(dataloader)
    return loss



def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update() 
            

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline



def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
            if it == 9: 
                break 

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


if __name__ == '__main__': 
    use_device = torch.cuda.is_available()
    device = torch.device('cuda' if use_device else 'cpu') 
    parser = argparse.ArgumentParser(description='Transformer Image Captioning')
    parser.add_argument('--features_path', type=str, default='/Users/feizhengcong/Desktop/COCO/features/coco_detections.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/Users/feizhengcong/Desktop/COCO/annotations') 
    parser.add_argument('--workers', type=int, default=0) 
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='Transformer')
    args = parser.parse_args()

    print(args)

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False) 
    
    # Create the dataset
    dataset = COCO(image_field, text_field, 'COCO/images/', args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits 
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb')) 
    
    # print(text_field.vocab.freqs) 
    # Model 
    encoder = VisualEncoder(3, 0, attention_module=ScaledDotProductAttention) 
    decoder = CaptionDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>']) 
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device) 
    
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()}) 
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    def lambda_lr(s):
        warm_up = 10000
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)

    # Initial conditions
    start_epoch = 0 
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98)) 
    scheduler = LambdaLR(optim, lambda_lr) 
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False 
    best_cider = .0
    patience = 0


    print('Training begins!')
    for e in range(start_epoch, start_epoch+80): 
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5) 
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=False,
                                           num_workers=args.workers)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
        else: 
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
        # Validation scores 
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        
        val_cider = scores['CIDEr']

        # Test scores
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True
        
        torch.save({
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'epoch': e,
                'state_dict': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'use_rl': use_rl,
            }, 'ckpt/%s_last.pth' % args.exp_name) 
        
        if best:
            copyfile('ckpt/%s_last.pth' % args.exp_name, 'ckpt/%s_best.pth' % args.exp_name)
        
        if exit_train: 
            break 
        






