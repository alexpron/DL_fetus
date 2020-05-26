import nibabel as nib
from os.path import expanduser

home = expanduser("~")
import os

import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
from PIL import Image
# import uuid
# import string, random
import csv
# import imageio
# from skimage import external
# import tifffile as tif

# put all the same type of image together in the same directory
src_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/data/'
dst_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/'
dst_dir3 = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/ribbon_space-T2w_dseg'

if not os.path.isdir(dst_directory):
    os.mkdir(dst_directory)

if not os.path.isdir(dst_dir3):
    os.mkdir(dst_dir3)

with open(src_directory+'participants.tsv', newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    data = list(reader)

variables = data[0]
data.pop(0)
birth_date = []

for row in data:
    vect = [row[0], row[2]]  # ["id", "birth_date"]
    birth_date.append(vect)


def take_second(elem):
    return elem[1]


birth_date_s = sorted(birth_date, key=take_second)
birth_date_100 = []
# count_100 = 0

for i in range(len(birth_date_s)):
    if float(birth_date_s[i][1]) < 36:
        # count_100 += 1
        birth_date_100.append(birth_date_s[i])

src_subjects = glob.glob(src_directory + 'sub-*', recursive=True)
num_cortex = 0

for subject in src_subjects:

    sub = subject.split('/')[9].split('-')[1]

    # select the subject that are in the list of the 100 youngest
    if any(sub in s for s in birth_date_100) and num_cortex < 150:

        src_sessions = glob.glob(subject + '/ses-*', recursive=True)

        for session in src_sessions:

            src_cortex = glob.glob(session + '/*-ribbon_space-T2w_dseg.nii.gz')
            if len(src_cortex) == 1:
                shutil.copy2(src_cortex[0], dst_dir3)
                num_cortex += 1

img_cortex = glob.glob(dst_dir3 + '/*-ribbon_space-T2w_dseg.nii.gz', recursive=True)

count = 0
num_imgs = 20

# indicate for which directory we want to generate masks
dirr = img_cortex

for j in range(len(dirr)):
    if count == num_imgs:
        break
    else:
        img = nib.load(dirr[j - 1])
        data = img.get_fdata()
        mask = np.random.rand(*data.shape)
        # id_ = uuid.uuid1()

        # lettersAndDigits = string.ascii_letters + string.digits
        # id_ = ''.join((random.choice(lettersAndDigits) for i in range(8)))

        for i in range(len(data[1, 1, :])):
            # values of the cortical plate
            mask[:, :, i] = (data[:, :, i] == 42)
            mask[:, :, i] += (data[:, :, i] == 3)

        # for i in range(len(data[1,1,:])):
        #    dst_dir_i = dst_directory+'ribbon_space-T2w_dseg/imgs/imgs'+str(j)+'/'
        #    dst_dir_m = dst_directory+'ribbon_space-T2w_dseg/mask/mask'+str(j)+'/'

        #    if not os.path.isdir(dst_dir_i):
        #        os.mkdir(dst_dir_i)
        #    if not os.path.isdir(dst_dir_m):
        #        os.mkdir(dst_dir_m)

        #    tif.imsave(dst_dir_i+'i'+"{0:0=3d}".format(i)+'.tif', data[:,:,i], bigtiff=True)
        #    tif.imsave(dst_dir_m+'m'+"{0:0=3d}".format(i)+'.tif', mask[:,:,i], bigtiff=True)
        #    count += 1

        data_ = data.astype(np.uint16)
        mask_ = mask.astype(np.uint16)

        # tif.imsave(dst_directory + 'ribbon_space-T2w_dseg/imgs/imgs_' + str(j) + '.tif', data_)
        # tif.imsave(dst_directory + 'ribbon_space-T2w_dseg/mask/mask_' + str(j) + '.tif', mask_)

        data_n = nib.Nifti1Image(data_, img.affine, img.header)
        mask_n = nib.Nifti1Image(mask_, img.affine, img.header)

        nib.save(data_n, dst_directory + 'ribbon_space-T2w_dseg/imgs/imgs_' + str(j) + '.nii.gz')
        nib.save(mask_n, dst_directory + 'ribbon_space-T2w_dseg/mask/mask_' + str(j) + '.nii.gz')

        count += 1