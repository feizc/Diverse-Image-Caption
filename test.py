import torch 
import os 
import json 
import pickle 
import clip 

annotation_path = 'COCO/annotations/captions_train2014.json' 
with open(annotation_path, 'r', encoding='utf-8') as f: 
    data_dict = json.load(f)['annotations']

print(data_dict[:10]) 

feature_saved_path = 'COCO/features/features.pickle' 
with open(feature_saved_path, 'rb') as f: 
    data_dict = pickle.load(f) 

print(data_dict) 

vocab_path = 'vocab_Transformer.pkl' 
with open(vocab_path, 'rb') as f:
    vocab_dict = pickle.load(f) 
print(vocab_dict)

text = clip.tokenize(["a diagram", "a dog", "a cat"])
print(text) 

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_features = model.encode_text(text) 
print(text_features.size())


