import torch 
import os 
import json 
import pickle 

annotation_path = 'data/annotations/captions_train2014.json' 
with open(annotation_path, 'r', encoding='utf-8') as f: 
    data_dict = json.load(f)['annotations']

print(data_dict[:10]) 

feature_saved_path = 'data/features/features.pickle' 
with open(feature_saved_path, 'rb') as f: 
    data_dict = pickle.load(f) 

print(data_dict)

