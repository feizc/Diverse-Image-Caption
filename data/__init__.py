from .field import RawField, ImageDetectionsField, TextField, ImagePatchField, CLIPTextField
from .dataset import COCO, ClipCOCO
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader(TorchDataLoader): 
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs) 