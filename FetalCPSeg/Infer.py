import os
import time

import nibabel as nib
import numpy as np

import torch

from Network import MixAttNet
from Utils import AvgMeter, check_dir
from DataOp import get_test_list, TrainGenerator


torch.cuda.set_device(0)

output_path = os.path.join('output_b32_nothing/')

test_list = get_test_list('../testset/fetus_test/')
net = MixAttNet().cuda()
net.load_state_dict(torch.load(output_path+'/ckpt/best_val.pth.gz'))

patch_size = 64
spacing = 4

save_path = os.path.join(output_path, 'save_data')
check_dir(save_path)

net.eval()

test_meter = AvgMeter()

for idx, data_dict in enumerate(test_list):
    image_path = data_dict['image_path']
    img_h = nib.load(image_path)
    image_ = img_h.get_fdata()
    image = TrainGenerator.preprocess(image_, np.nanmin(image_)).squeeze()
    name = image_path.split('/')[3]

    #mask_path = data_dict['label_path']
    #mask_ = nib.load(mask_path).get_fdata()
    #mask = TrainGenerator.preprocess(mask_, 0).squeeze()
    w, h = image.shape

    pre_count = np.zeros_like(image, dtype=np.float32)
    predict = np.zeros_like(image, dtype=np.float32)

    x_list = np.squeeze(np.concatenate((np.arange(0, w - patch_size, patch_size // spacing)[:, np.newaxis],
                                        np.array([w - patch_size])[:, np.newaxis])).astype(np.int))
    y_list = np.squeeze(np.concatenate((np.arange(0, h - patch_size, patch_size // spacing)[:, np.newaxis],
                                        np.array([h - patch_size])[:, np.newaxis])).astype(np.int))
    start_time = time.time()

    for x in x_list:
        for y in y_list:
            image_patch = image[x:x + patch_size, y:y + patch_size].astype(np.float32)
            patch_tensor = torch.from_numpy(image_patch[np.newaxis, np.newaxis]).cuda()
            predict[x:x + patch_size, y:y + patch_size] += net(patch_tensor).squeeze().cpu().data.numpy()
            pre_count[x:x + patch_size, y:y + patch_size] += 1

    predict /= pre_count

    predict = np.squeeze(predict)
    image = np.squeeze(image)

    predict[predict > 0.5] = 1
    predict[predict < 0.5] = 0

    #image_nii = nib.Nifti1Image(image, affine=None)
    predict_nii = nib.Nifti1Image(predict, affine=None)
    #mask_nii = nib.Nifti1Image(mask, affine=None)

    nib.save(predict_nii, os.path.join(save_path, name))
    #check_dir(os.path.join(save_path, '{}'.format(idx)))
    #nib.save(image_nii, os.path.join(save_path, '{}/image.nii.gz'.format(idx)))
    #nib.save(predict_nii, os.path.join(save_path, '{}/predict.nii.gz'.format(idx)))
    #nib.save(mask_nii, os.path.join(save_path, '{}/label.nii.gz'.format(idx)))

    print("[{}] Testing Finished, Cost {:.2f}s".format(idx, time.time()-start_time))
