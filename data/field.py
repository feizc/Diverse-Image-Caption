from torch.functional import split
from torch.utils.data.dataloader import default_collate 
from collections import Counter, OrderedDict 
from itertools import chain 
import torch 
import os  
import shutil 
import warnings 
import h5py 
import numpy as np 
import six 
from PIL import Image
from torchvision import transforms 
import PIL 
import importlib.util 
from typing import Any, Optional, Tuple, List, Union
import clip 

from .vocab import Vocab 
from .utils import get_tokenizer 
from .dataset import Dataset 


class RawField(object): 
    '''a general datatype'''
    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing 
    
    def preprocess(self, x):
        if self.preprocessing is not None: 
            return self.preprocessing(x) 
        else:
            return x 
    
    def postprocess(self, batch, *args, **kwargs): 
        '''list -> batch'''
        if self.postprocessing is not None:
            batch = self.postprocessing(batch) 
        return default_collate 
    
    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)



class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=100,
                 sort_by_prob=False, load_in_tmp=True):
        self.max_detections = max_detections 
        self.detections_path = detections_path 
        self.sort_by_prob = sort_by_prob 

        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path)) 

        if load_in_tmp: 
            if not os.path.isfile(tmp_detections_path):
                print('free: ', shutil.disk_usage('/tmp')[-1], 
                        'will use: ', os.path.getsize(detections_path))
                if shutil.disk_usage('/tmp')[-1] < os.path.getsize(detections_path): 
                    warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
            else:
                self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)
    
    def preprocess(self, x):
        image_id = int(x.split('_')[-1].split('.')[0]) 
        try:
            f = h5py.File(self.detections_path, 'r')
            precomp_data = f['%d_features' % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[np.argsort(np.max(f['%d_cls_prob' % image_id][()], -1))[::-1]]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10,2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32)



class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=100,
                 sort_by_prob=False, load_in_tmp=True):
        self.max_detections = max_detections 
        self.detections_path = detections_path 
        self.sort_by_prob = sort_by_prob 

        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path)) 

        if load_in_tmp: 
            if not os.path.isfile(tmp_detections_path):
                print('free: ', shutil.disk_usage('/tmp')[-1], 
                        'will use: ', os.path.getsize(detections_path))
                if shutil.disk_usage('/tmp')[-1] < os.path.getsize(detections_path): 
                    warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
            else:
                self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)
    
    def preprocess(self, x):
        image_id = int(x.split('_')[-1].split('.')[0]) 
        try:
            f = h5py.File(self.detections_path, 'r')
            precomp_data = f['%d_features' % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[np.argsort(np.max(f['%d_cls_prob' % image_id][()], -1))[::-1]]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10,2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32)


class ImagePatchField(RawField): 
    def __init__(self, image_file_path=None, preprocessing=None, postprocessing=None, transform=None): 
        self.image_file_path = image_file_path 
        self.image_extractor = transform
        super(ImagePatchField, self).__init__(preprocessing, postprocessing) 
    def preprocess(self, x):
        image_path = os.path.join(self.image_file_path, x) 
        image_PIL = Image.open(image_path) 
        return self.image_extractor(image_PIL).squeeze(0)



def is_torch_tensor(obj):
    return isinstance(obj, torch.Tensor) if importlib.util.find_spec("torch") is not None else False


class ImageExtractor: 
    """
    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int`, `optional`, defaults to 224):
            Resize the input to the given size. Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BOX`,
            :obj:`PIL.Image.BILINEAR`, :obj:`PIL.Image.HAMMING`, :obj:`PIL.Image.BICUBIC` or :obj:`PIL.Image.LANCZOS`.
            Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        do_center_crop (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to crop the input at the center. If the input size is smaller than :obj:`crop_size` along any edge,
            the image is padded with 0's and then center cropped.
        crop_size (:obj:`int`, `optional`, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if :obj:`do_center_crop` is set to
            :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with :obj:`image_mean` and :obj:`image_std`.
        image_mean (:obj:`List[int]`, defaults to :obj:`[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (:obj:`List[int]`, defaults to :obj:`[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """
    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BICUBIC,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=False,
        image_std=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  
        ],
        **kwargs
    ):
        """
        Main method to prepare for the model one or several image(s).
        .. warning::
           NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
           PIL images.
        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width. 
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        images = np.array(images)
        # return as BatchFeature
        return images 

    def center_crop(self, image, size):
        """
        Crops :obj:`image` to the given size using a center crop. Note that if the image is too small to be cropped to
        the size is given, it will be padded (so the returned result has the size asked).
        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to resize.
            size (:obj:`int` or :obj:`Tuple[int, int]`):
                The size to which crop the image.
        """
        if not isinstance(size, tuple):
            size = (size, size)

        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)

        image_width, image_height = image.size
        crop_height, crop_width = size

        crop_top = int((image_height - crop_height + 1) * 0.5)
        crop_left = int((image_width - crop_width + 1) * 0.5)

        return image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

    def resize(self, image, size, resample=Image.BICUBIC):
        """
        Resizes :obj:`image`. Note that this will trigger a conversion of :obj:`image` to a PIL Image.
        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to resize.
            size (:obj:`int` or :obj:`Tuple[int, int]`):
                The size to use for resizing the image. If :obj:`int` it will be resized to match the shorter side
            resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
                The filter to user for resampling.
        """

        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)
        if isinstance(size, tuple):
            new_w, new_h = size
        else:
            width, height = image.size
            short, long = (width, height) if width <= height else (height, width)
            if short == size:
                return image
            new_short, new_long = size, int(size * long / short)
            new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
        return image.resize((new_w, new_h), resample) 
    
    def normalize(self, image, mean, std):
        """
        Normalizes :obj:`image` with :obj:`mean` and :obj:`std`. Note that this will trigger a conversion of
        :obj:`image` to a NumPy array if it's a PIL Image.
        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to normalize.
            mean (:obj:`List[float]` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (:obj:`List[float]` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
        """

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)
        elif is_torch_tensor(image):
            import torch

            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(std, torch.Tensor):
                std = torch.tensor(std)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None] 
        elif image.ndim == 3 and image.shape[0] > 3: 
            return (image[:3, :, :] - mean[:, None, None]) / std[:, None, None]
        elif image.ndim == 2: 
            image = np.expand_dims(image, 0).repeat(3, axis=0)
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std 
        
    def to_numpy_array(self, image, rescale=None, channel_first=True):
        """
        Converts :obj:`image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.
        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (:obj:`bool`, `optional`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to :obj:`True` if the image is a PIL Image or an array/tensor of integers, :obj:`False`
                otherwise.
            channel_first (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if is_torch_tensor(image):
            image = image.numpy()

        if rescale is None:
            rescale = isinstance(image.flat[0], np.integer)

        if rescale:
            image = image.astype(np.float32) / 255.0

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image



class TextField(RawField): 
    vocab_cls = Vocab 
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        if nopoints:
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing) 

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x 
    
    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)
        print('sources: ', sources)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs) 
    

    def process(self, batch, device=None):
        padded = self.pad(batch) 
        #print(padded)
        tensor = self.numericalize(padded, device=device)
        #print(tensor)
        return tensor 
    
    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions


class CLIPTextField(RawField): 
    vocab_cls = Vocab 
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"] 

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        if nopoints:
            self.punctuations.append("..")

        super(CLIPTextField, self).__init__(preprocessing, postprocessing) 

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x 
    
    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)
        print('sources: ', sources)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs) 
    

    def process(self, batch, device=None): 
        #print(batch) 
        s_list = self._to_list(batch) 
        txt = clip.tokenize(s_list)
        #print(txt)
        #padded = self.pad(batch) 
        #tensor = self.numericalize(padded, device=device)
        #print(tensor)
        return txt
    
    def _to_list(self, batch): 
        batch = list(batch) 
        sentence_list = [] 
        for s_list in batch: 
            s = '' 
            for w in s_list:
                s += ' ' + w 
            sentence_list.append(s) 
        return sentence_list

    
    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions



