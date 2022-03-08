# Diverse-Image-Caption

Promoting Coherence and Diversity in Image Captioning

This repository includes the reference code for conventional diverse image captioning models and CLIP-CVAE.  

## Environment setup 

- Python 3.8
- Pytorch 1.9
- transformers 4.12 

## Data preparation
To run the code, annotations and images for the COCO dataset are needed.
Please download the zip files including the images ([train2014.zip](http://images.cocodataset.org/zips/train2014.zip), [val2014.zip](http://images.cocodataset.org/zips/val2014.zip)),
the zip file containing the annotations ([annotations_trainval2014.zip](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)) and extract them. These paths will be set as arguments later. 

## Training 
Run `python train_CVAE.py` using the following arguments: 

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name (default: `ClipCVAE`) | 
| `--batch_size` | Batch size (default: `25`) | 
| `--workers` | Number of workers (default: `0`) | 
| `--annotation_folder` | Path to folder with COCO annotations (required) | 
| `--image_folder` | Path to folder with COCO images (required) | 
| `--warmup` | Warmup value for learning rate scheduling (default: `10000`) | 
| `--gpt_path` | Path to folder with GPT2 model (required) | 




## Acknowledgment
This repository refers to [github](https://github.com/aimagelab/meshed-memory-transformer) and [huggingface](https://github.com/huggingface/transformers). 
Thanks for the released  code. 

