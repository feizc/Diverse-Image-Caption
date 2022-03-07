# extract image features with faster-rcnn
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
from torchvision.models import detection 
from tqdm import tqdm 


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def image_feature_extract(image_data_path, feature_saved_path):
    image_name_list = os.listdir(image_data_path) 
    
    use_device = torch.cuda.is_available()
    device = torch.device('cuda' if use_device else 'cpu') 
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
    feature_extractor = feature_extractor.to(device)

    detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    detection_model.eval() 
    detection_model.to(device)
    image_features_dict = {}
    
    for image_name in tqdm(image_name_list): 
        image_path = os.path.join(image_data_path, image_name) 
        image_PIL = Image.open(image_path).convert('RGB') 
        width, height = image_PIL.size
        # image_PIL.show() 
        # https://blog.csdn.net/tsq292978891/article/details/78767326 
        image_tensor = transform(image_PIL).to(device) 
        
        # https://blog.csdn.net/yanxiangtianji/article/details/112256618
        predictions = detection_model([image_tensor]) 
        boxes = predictions[0]['boxes']
        # plot the extracted objective 
        # draw = ImageDraw.Draw(image_PIL) 
        #for i in range(10):
        #    draw.rectangle(boxes[i].tolist(),outline=(255,0,0)) 
        #image_PIL.show() 
        
        image_feature_list = []

        for box in boxes[:5]: 
            box = box.tolist() 
            x11 = min(box[0], box[2])
            x22 = max(box[0], box[2]) 
            y11 = min(box[1], box[3]) 
            y22 = max(box[1], box[3]) 

            x1 = max(int(x11), 0) 
            y1 = max(int(y11), 0)
            x2 = min(int(x22), width)
            y2 = min(int(y22), height)
            if x1 == x2 or y1 == y2: 
                continue 
            image_crop = image_PIL.crop(tuple([x1, y1, x2, y2])) 
            # image_crop.show()
            image_crop = crop_transform(image_crop).to(device) 
            image_crop.unsqueeze_(0)
            image_feature = feature_extractor(image_crop).squeeze(2).squeeze(2)[0] 
            image_feature_list.append(image_feature.tolist()) 
        image_features = torch.Tensor(image_feature_list) 
        image_features_dict[image_name] = image_features  
        break 
    
    with open(os.path.join(feature_saved_path, 'features.pickle'), 'wb') as fp: 
        pickle.dump(image_features_dict, fp) 


if __name__ == '__main__': 
    train_image_path = 'COCO/images/train2014'  
    feature_saved_path = 'COCO/features'
    image_feature_extract(train_image_path, feature_saved_path)
