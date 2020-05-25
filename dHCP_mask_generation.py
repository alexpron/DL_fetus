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
import string, random
import csv

# put all the same type of image together in the same directory
src_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/dhcp_t2_and_seg_data/'
dst_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/'
# dst_dir0 = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/restore_T2w'
# dst_dir1 = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/drawem87_space-T2w_dseg'
# dst_dir2 = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/drawem9_space-T2w_dseg'
dst_dir3 = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/ribbon_space-T2w_dseg'

if not os.path.isdir(dst_directory):
    os.mkdir(dst_directory)

# if not os.path.isdir(dst_dir0):
#    os.mkdir(dst_dir0)

# if not os.path.isdir(dst_dir1):
#    os.mkdir(dst_dir1)

# if not os.path.isdir(dst_dir2):
#    os.mkdir(dst_dir2)

if not os.path.isdir(dst_dir3):
    os.mkdir(dst_dir3)

with open(home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dhcp_t2_and_seg_data/participants.tsv', newline='') as tsvfile:
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

while num_cortex <= 100:
    for subject in src_subjects:
        sub = subject.split('/')[9].split('-')[1]

        # select the subject that are in the list of the 100 youngest
        if any(sub in s for s in birth_date_100):
            src_sessions = glob.glob(subject + '/ses-*', recursive=True)

            for session in src_sessions:
                # src_t2w = glob.glob(session+'/*-restore_T2w.nii.gz')[0]
                # shutil.copy2(src_t2w, dst_dir0)

                # src_drawem87 = glob.glob(session+'/*-drawem87_space-T2w_dseg.nii.gz')[0]
                # shutil.copy2(src_drawem87, dst_dir1)

                # src_drawem9 = glob.glob(session+'/*-drawem9_space-T2w_dseg.nii.gz')[0]
                # shutil.copy2(src_drawem9, dst_dir2)

                src_cortex = glob.glob(session + '/*-ribbon_space-T2w_dseg.nii.gz')
                if len(src_cortex) == 1:
                    shutil.copy2(src_cortex[0], dst_dir3)
                    num_cortex += 1

# img_T2w = glob.glob(dst_dir0+'/*-restore_T2w.nii.gz', recursive=True)
# img_drawem87 = glob.glob(dst_dir1+'/*-drawem87_space-T2w_dseg.nii.gz', recursive=True)
# img_drawem9 = glob.glob(dst_dir2+'/*-drawem9_space-T2w_dseg.nii.gz', recursive=True)
img_cortex = glob.glob(dst_dir3 + '/*-ribbon_space-T2w_dseg.nii.gz', recursive=True)

count = 0
num_imgs = 100  # len(img_cortex)

# indicate for which directory we want to generate masks
dirr = img_cortex

for j in range(len(dirr)):
    if count == num_imgs:
        break
    else:
        img0 = nib.load(dirr[j])
        data0 = img0.get_fdata()
        mask0 = np.random.rand(*data0.shape)
        # id_ = uuid.uuid1()

        lettersAndDigits = string.ascii_letters + string.digits
        id_ = ''.join((random.choice(lettersAndDigits) for i in range(8)))

        for i in range(len(data0[1, 1, :])):
            mask0[:, :, i] = (data0[:, :, i] > 10)

        final_m = mask0[:, :, 100]
        final_d = data0[:, :, 100]

        # Image.fromarray assumes the input is laid-out as unsigned 8-bit integers
        final_m = (final_m * 255).astype(np.uint8)
        mask_img = Image.fromarray(final_m, mode='L')
        mask_img.save(dst_directory + 'ribbon_space-T2w_dseg/mask/' + str(id_) + '_' + str(j) + '.jpg')

        # plt.imsave(dst_directory+'/mask/data'+str(j)+'.jpg', final_m)
        plt.imsave(dst_directory + 'ribbon_space-T2w_dseg/imgs/' + str(id_) + '_' + str(j) + '.jpg',
                   final_d)  # cmap = 'gray'

        count += 1
