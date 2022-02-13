import random
from sys import prefix 
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.functional import batch_norm 
from torch.optim import Adam 
from torch.optim.lr_scheduler import LambdaLR 
from torch.nn import NLLLoss 
import clip 

import argparse, os, pickle 
import numpy as np
from tqdm import tqdm 
from torch.utils.data import dataloader
from transformers.utils.dummy_tokenizers_objects import GPT2TokenizerFast 
from model.transformer import Transformer 
import multiprocessing
import itertools 
from shutil import copyfile 
from torch.utils.data import DataLoader

from data import RawCOCO
from model import ClipVAE
from transformers import GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import evaluation 
from evaluation import PTBTokenizer, Cider
from torch.nn import functional as nnf 
import warnings
warnings.filterwarnings("ignore")


random.seed(2021)
torch.manual_seed(2021)
np.random.seed(2021)


def compute_loss(model, image_features, text_features, captions, mask, beta=1.0):
    outputs = model(captions, image_features, text_features,) 
    logits = outputs[0].logits[:,  4: -1]
    kl_loss = outputs[-1] 
    num_logits = logits.size(-1) 

    if mask is not None: 
        mask = mask.type(torch.bool) 
        mask = mask.to(device) 
        logits = logits.masked_select(mask.unsqueeze(-1)) 
        target_tokens = captions.masked_select(mask) 
    
    ce_loss = nnf.cross_entropy(logits.view(-1, num_logits), target_tokens.view(-1), ignore_index=0) 
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss 
    return loss, ce_loss, kl_loss 



def train(model, dataloader, optim, scheduler): 
    model.train()
    running_loss = .0 
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar: 
        for it, (image_features, text_features, tokens, mask) in enumerate(dataloader): 
            image_features, text_features, tokens, mask = image_features.to(device), text_features.to(device), \
                                                            tokens.to(device), mask.to(device)

            loss, ce_loss, kl_loss = compute_loss(model, image_features, text_features, tokens, mask, 1.0) 
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()

            this_loss = loss.item()
            running_loss += this_loss
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update() 
            break 
            
    loss = running_loss / len(dataloader)
    return loss



def generate_sample(model, tokenizer, prefix_emb, top_p=0.8, temperature=1.0): 
    generated_list = [] 
    stop_token_index = tokenizer.encode('<|endoftext|>') 
    max_length = 30  
    filter_value = -float("Inf") 
    tokens = None
    with torch.no_grad(): 
        generated = prefix_emb 
        for idx in range(max_length): 
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True) 
            cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token) 
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break

        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list)
        generated_list.append(output_list)

    return generated_list[0]


def evaluate_metrics(model, dataloader, tokenizer):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (image_features, text_features, tokens, mask) in enumerate(iter(dataloader)):
            image_features, text_features, tokens, mask = image_features.to(device), text_features.to(device), \
                                                            tokens.to(device), mask.to(device)
            prefix_features = torch.cat((image_features, text_features), dim=1)
            posterior_mean, posterior_logvar = model.posternet(prefix_features)
            gen_s = generate_sample(model, tokenizer, posterior_mean.view(-1, 5, 768)) 
            gts[it] = [tokenizer.decode(tokens[0].tolist())]
            gen[it] = [tokenizer.decode(gen_s)]  
            pbar.update()
            break

    scores, _ = evaluation.compute_scores(gts, gen) 
    print(scores)
    return scores


if __name__ == '__main__': 
    use_device = torch.cuda.is_available()
    device = torch.device('cuda:2' if use_device else 'cpu') 
    parser = argparse.ArgumentParser(description='Transformer Image Captioning')
    parser.add_argument('--features_path', type=str, default='/Users/feizhengcong/Desktop/COCO/features/coco_detections.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/Users/feizhengcong/Desktop/COCO/annotations/captions_train2014.json') 
    parser.add_argument('--image_path', type=str, default='/Users/feizhengcong/Desktop/COCO/images')
    parser.add_argument('--gpt_path', type=str, default='ckpt/gpt2')
    parser.add_argument('--workers', type=int, default=0) 
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='ClipVAE')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    print(args)

    # load clip and image preprocess operation 
    clip_model, img_trans = clip.load("ViT-B/32", device=device) 
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_path)
    
    dataset = RawCOCO(args.annotation_folder, args.image_path, clip_model, img_trans, tokenizer, device)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
    model = ClipVAE().to(device) 
    optimizer = AdamW(model.parameters(), lr=2e-5) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5000, num_training_steps=args.epochs * len(train_dataloader)
    )
    
    print('Training begins!') 
    for e in range(args.epochs): 
        train(model, train_dataloader, optimizer, scheduler)
        evaluate_metrics(model, eval_dataloader, tokenizer)

