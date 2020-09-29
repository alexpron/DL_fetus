from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
import torchvision.transforms.functional as TF
import nibabel as nib


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = 1
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(splitext(file)[0])[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_arr, fill_value):
        img = img_arr.copy()
        img[np.isnan(img)] = fill_value
        im_t = TF.to_tensor(img)
        image = im_t.cpu().numpy()  # (1, 290, 290)

        return image

    @classmethod
    def noise_injection(cls, image):
        noise = random.random()
        if noise < 0.3:
            # Gaussian noise
            std = random.uniform(0,0.5)
            mean = 0
            gaussian = np.random.normal(loc=mean, scale=std, size=image.shape)
            noisy_image = image + gaussian
        elif 0.3 <= noise < 0.6:
            # Salt & Pepper noise
            s_vs_p = 0.5
            amount = 0.005
            noisy_image = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            for s in range(len(coords[0])):
                if ~np.isnan(noisy_image[coords[0][s]][coords[1][s]]):
                    noisy_image[coords[0][s]][coords[1][s]] = 1
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            for p in range(len(coords[0])):
                if ~np.isnan(noisy_image[coords[0][p]][coords[1][p]]):
                    noisy_image[coords[0][p]][coords[1][p]] = 0
        elif 0.6 <= noise < 0.8:
            # Speckle noise
            row,col = image.shape
            gauss = np.random.randn(row,col)/3
            noisy_image = image + image * gauss
        else:
            noisy_image = image

        return noisy_image

    @classmethod
    def get_augmentation(cls, image, mask, min_pix_img, min_pix_mask):

        im_pil = Image.fromarray(image[0])
        mk_pil = Image.fromarray(mask[0])

        r = 10
        degrees = random.randrange(-r, r)
        param = random.random()
        w = random.randrange(-r, r)
        h = random.randrange(-r, r)

        pil_img_fill = min_pix_img
        pil_msk_fill = min_pix_mask

        if param <= 0.3:
            # print('Translation')
            im = TF.affine(im_pil, angle=0, translate=(w, h),
                           resample=Image.NEAREST, scale=1, shear=0, fillcolor=pil_img_fill)
            mk = TF.affine(mk_pil, angle=0, translate=(w, h),
                           resample=Image.NEAREST, scale=1, shear=0, fillcolor=pil_msk_fill)
        elif 0.3 < param < 0.6:
            # print('Rotation')
            im = TF.affine(im_pil, angle=degrees, translate=(0, 0),
                           resample=Image.NEAREST, scale=1, shear=0, fillcolor=pil_img_fill)
            mk = TF.affine(mk_pil, angle=degrees, translate=(0, 0),
                           resample=Image.NEAREST, scale=1, shear=0, fillcolor=pil_msk_fill)
        else:
            # print('Nothing')
            im = im_pil
            mk = mk_pil

        im_t = TF.to_tensor(im)
        img_aug = im_t.cpu().numpy()
        mk_t = TF.to_tensor(mk)
        msk_aug = mk_t.cpu().numpy()

        return img_aug, msk_aug

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.nii.gz')
        img_file = glob(self.imgs_dir + idx + '.nii.gz')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = nib.load(img_file[0]).get_fdata()
        mask = nib.load(mask_file[0]).get_fdata()

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        noisy_image = self.noise_injection(img)
        # noisy_image = img
        img_fill = np.nanmin(noisy_image)
        mask_fill = 0
        img_ = self.preprocess(noisy_image, img_fill)
        mask_ = self.preprocess(mask, mask_fill)
        image, label = self.get_augmentation(img_, mask_, img_fill, mask_fill)

        return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(label)}
