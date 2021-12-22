import torch 
import torchvision
import os 
from PIL import Image, ImageDraw
from torchvision import transforms 
import copy 
import torchvision.models as models
import torch.nn as nn
import numpy as np 
import pickle 
from tqdm import tqdm 


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def image_feature_extract(image_data_path, feature_saved_path):
    image_name_list = os.listdir(image_data_path) 
    
    transform = transforms.ToTensor() 
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    crop_transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    resnext50_32x4d = models.resnext101_32x8d(pretrained=True) 
    feature_extractor = nn.Sequential(*(list(resnext50_32x4d.children())[:-1]))

    image_features_dict = {}
    
    for image_name in tqdm(image_name_list): 
        image_path = os.path.join(image_data_path, image_name) 
        image_PIL = Image.open(image_path).convert('RGB') 
        # image_PIL.show() 
        # https://blog.csdn.net/tsq292978891/article/details/78767326 
        image_tensor = transform(image_PIL) 
        # https://blog.csdn.net/yanxiangtianji/article/details/112256618
        detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
        detection_model.eval() 
        predictions = detection_model([image_tensor]) 
        boxes = predictions[0]['boxes']
        # plot the extracted objective 
        # draw = ImageDraw.Draw(image_PIL) 
        #for i in range(10):
        #    draw.rectangle(boxes[i].tolist(),outline=(255,0,0)) 
        #image_PIL.show() 
        
        image_feature_list = []

        for box in boxes[:2]: 
            image_crop = image_PIL.crop(tuple(box.tolist()))
            image_crop = crop_transform(image_crop)
            image_crop.unsqueeze_(0)
            image_feature = feature_extractor(image_crop).squeeze(2).squeeze(2)[0] 
            image_feature_list.append(image_feature.tolist()) 
        image_features = torch.Tensor(image_feature_list) 
        image_features_dict[image_name] = image_features  
        break 
    
    with open(os.path.join(feature_saved_path, 'features.pickle'), 'wb') as fp: 
        pickle.dump(image_features_dict, fp) 


if __name__ == '__main__': 
    train_image_path = 'data/images/train2014'  
    feature_saved_path = 'data/features'
    image_feature_extract(train_image_path, feature_saved_path)
