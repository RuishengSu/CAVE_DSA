import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import dicom_reader.reader as dicom_reader
from skimage.transform import resize
import nibabel as nib


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', transform=None):
        self.transform = transform
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in listdir(masks_dir) if not file.startswith('.')]
        self.ids = [i for i in self.ids if i in self.mask_ids]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, pil_mask, scale, transform=None):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        pil_mask = pil_mask.resize((newW, newH), resample=Image.NEAREST)
        img_ndarray, mask_ndarray = np.asarray(pil_img), np.asarray(pil_mask)

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255

        return img_ndarray, mask_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return torch.load(filename).numpy()
        elif ext in ['.dcm']:
            ds = dicom_reader.read_file(filename, defer_size="1 KB", stop_before_pixels=False, force=True)
            return ds.pixel_array
        elif ext in ['.nii']:
            img_obj = nib.load(filename)
            return np.transpose(img_obj.get_fdata(), (1, 0, 2))
        else:
            return np.asarray(Image.open(filename))

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.shape[:2] == mask.shape, \
            'Image and mask {} should be the same size, but are {} and {}'.format(name, img.shape[:2], mask.shape)

        img, mask = self.preprocess(img, mask, self.scale, transform=self.transform)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'filename': name
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')


class DSADataset(BasicDataset):

    @classmethod
    def preprocess(cls, img, mask, scale, transform=None):
        if transform is not None:
            transformed = transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]
        w, h = img.shape[:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        mask = resize(mask, (newW, newH), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
            img = resize(img, (img.shape[0], newW, newH), anti_aliasing=False, preserve_range=True)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = resize(img, (img.shape[0], newW, newH), anti_aliasing=False, preserve_range=True)
            img = img[:, np.newaxis, ...]
        img = img / 255

        return img, mask
