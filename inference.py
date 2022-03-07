import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from model import Transformer, VisualEncoder, CaptionDecoder, ScaledDotProductAttention
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import json 

import warnings
warnings.filterwarnings("ignore")

random.seed(2022)
torch.manual_seed(2022)
np.random.seed(2022)


def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {} 
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False) 
            
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
            break 
    # discard point 
    #gts = evaluation.PTBTokenizer.tokenize(gts)
    #gen = evaluation.PTBTokenizer.tokenize(gen) 
    scores, _ = evaluation.compute_scores(gts, gen) 
    return scores


if __name__ == '__main__':
    use_device = torch.cuda.is_available()
    device = torch.device('cuda' if use_device else 'cpu') 

    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str, default='/Users/feizhengcong/Desktop/COCO/features/coco_detections.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='/Users/feizhengcong/Desktop/COCO/annotations')
    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab_Transformer.pkl', 'rb'))

    # Model and dataloaders
    encoder = VisualEncoder(3, 0, attention_module=ScaledDotProductAttention) 
    decoder = CaptionDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>']) 
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device) 

    data = torch.load('ckpt/Transformer_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores) 
    