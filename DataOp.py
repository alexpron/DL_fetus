import os
import glob

import nibabel as nib
import numpy as np
from PIL import Image

from os import listdir
from os.path import splitext

import torchvision.transforms.functional as TF
import torch
from torch.utils.data import Dataset

from volumentations import *


def get_list(imgs_dir, mask_dir):
    """
    This function is to read data from data dir.
    The data dir should be set as follow:
    -- Data
        -- case1
            -- image.nii.gz
            -- label.nii.gz
        -- case2
        ...
    """
    print("Reading Data...")

    ids = [splitext(splitext(file)[0])[0] for file in listdir(imgs_dir) if not file.startswith('.')]

    dict_list = []

    for idx in ids:
        dict_list.append(
            {
                'image_path': os.path.join(imgs_dir, idx+'.nii.gz'),
                'label_path': os.path.join(mask_dir, idx+'.nii.gz'),
            }
        )

    # we split the data set to train set(0.75), val set(0.05), test set(0.2)
    train_ratio = 0.75
    val_ratio = 0.8
    train_num = round(len(dict_list)*train_ratio)
    val_num = round(len(dict_list)*val_ratio)

    train_list = dict_list[:train_num] + dict_list[:train_num]
    val_list = dict_list[train_num:val_num]
    test_list = dict_list[val_num:]
    print("Finished! Train:{} Val:{} Test:{}".format(len(train_list), len(val_list), len(test_list)))

    return train_list, val_list, test_list


def get_test_list(imgs_dir):
    print("Reading Data...")

    input_files = sorted(glob.glob(imgs_dir + '*.nii.gz'))
    #input_GT = sorted(glob.glob(imgs_dir + 'GT/*.nii.gz'))

    dict_list = []

    # if len(input_files) == len(input_GT):
    # for path in path_list:
    for i in range(len(input_files)):
        # print(idx)
        dict_list.append(
            {
                'image_path': input_files[i]  # os.path.join(imgs_dir, idx+'.nii.gz'),
                #'label_path': input_GT[i],  # os.path.join(mask_dir, idx+'.nii.gz'),
            }
        )

    # we split the data set to train set(0.75), val set(0.05), test set(0.2)
    # print("Finished! Train:{} Val:{} Test:{}".format(len(train_list), len(val_list), len(test_list)))

    return dict_list


class TrainGenerator(object):
    """
    This is the class to generate the patches
    """

    def __init__(self, data_list, batch_size, patch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        self.patch_size = patch_size
    
    @classmethod
    def noise_injection(cls, image):
        noise = random.random()
        if noise < 0.3:
            # Gaussian noise
            std = random.uniform(0, 0.5)
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
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            for p in range(len(coords[0])):
                if ~np.isnan(noisy_image[coords[0][p]][coords[1][p]]):
                    noisy_image[coords[0][p]][coords[1][p]] = 0
        elif 0.6 <= noise < 0.8:
            # Speckle noise
            row, col = image.shape
            gauss = np.random.randn(row, col) / 3
            # gauss = gauss.reshape(row,col)
            noisy_image = image + image * gauss
        else:
            noisy_image = image

        return noisy_image

    @classmethod
    def get_all_augmentation(cls, image, mask, min_pix_img, min_pix_mask):

        im_pil = Image.fromarray(image)
        mk_pil = Image.fromarray(mask)

        r = 10
        degrees = random.randrange(-r, r)
        param = random.random()
        w = random.randrange(-r, r)
        h = random.randrange(-r, r)

        pil_img_fill = min_pix_img
        pil_msk_fill = min_pix_mask

        if param <= 0.3:
            im = TF.affine(im_pil, angle=0, translate=(w, h),
                           resample=Image.NEAREST, scale=1, shear=0, fillcolor=pil_img_fill)
            mk = TF.affine(mk_pil, angle=0, translate=(w, h),
                           resample=Image.NEAREST, scale=1, shear=0, fillcolor=pil_msk_fill)
        elif 0.3 < param < 0.6:
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

        #print('Image aug: '+str(img_aug.shape))
        #print('Img Squeeze: '+str(img_aug.shape.squeeze()))
        #print('Msk aug: '+str(msk_aug.shape))
        #print('Msk Squeeze: '+str(msk_aug.shape.squeeze()))
        return img_aug.squeeze(), msk_aug.squeeze()

    @classmethod
    def preprocess(cls, img_arr, fill_value):
        img = img_arr.copy()
        img[np.isnan(img)] = fill_value
        im_t = TF.to_tensor(img)
        image = im_t.cpu().numpy()

        return image

    def get_item(self):

        dict_list = random.sample(self.data_list, self.batch_size)

        image_list = [dict_item['image_path'] for dict_item in dict_list]
        label_list = [dict_item['label_path'] for dict_item in dict_list]

        image_patch, label_patch = self._sample_patch(image_list, label_list)
        print(image_patch.shape)
        print(type(image_patch))

        return image_patch, label_patch

    def _sample_patch(self, image_list, clean_list):
        half_size = self.patch_size // 2
        image_patch_list = []
        label_patch_list = []

        for image_path, clean_path in zip(image_list, clean_list):
            image_ = nib.load(image_path).get_fdata()
            label_ = nib.load(clean_path).get_fdata()

            noisy_image = self.noise_injection(image_)

            img = self.preprocess(noisy_image, np.nanmin(noisy_image)).squeeze()
            msk = self.preprocess(label_, 0).squeeze()
            # (290,290)
            #print('Image: '+str(img.shape) +'\t Mask: '+str(msk.shape))
            image, label = self.get_all_augmentation(img, msk, np.nanmin(noisy_image), 0)
            #print('Image aug: '+str(image.shape) +'\t Mask aug: '+str(label.shape))
            # here we augment the corresponding data and label
            #data = {'image': img, 'label': msk}
            #aug_data = self.aug(**data)
   
            #image, label = aug_data['image'], aug_data['label']
            w, h = image.shape

            label_index = np.where(label == 1)
            length_label = label_index[0].shape[0]

            p = random.random()
            # we set a probability(p) to make most of the center of sampling patches
            # locate to the regions with label not background
            if p < 0.875 and length_label > 1:
                sample_id = random.randint(1, length_label-1)
                x, y = label_index[0][sample_id], label_index[1][sample_id]
            else:
                x, y = random.randint(0, w), random.randint(0, h)

            # here we prevent the sampling patch overflow volume
            if x < half_size:
                x = half_size
            elif x > w-half_size:
                x = w-half_size-1

            if y < half_size:
                y = half_size
            elif y > h-half_size:
                y = h-half_size-1

            image_patch = image#[x-half_size:x+half_size, y-half_size:y+half_size].astype(np.float32)
            label_patch = label#[x-half_size:x+half_size, y-half_size:y+half_size].astype(np.float32)

            image_patch_list.append(image_patch[np.newaxis, np.newaxis])
            label_patch_list.append(label_patch[np.newaxis, np.newaxis])

        image_out = np.concatenate(image_patch_list, axis=0)
        label_out = np.concatenate(label_patch_list, axis=0)

        return image_out, label_out

