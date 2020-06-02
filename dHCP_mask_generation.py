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
# import tifffile as tif

# put all the same type of image together in the same directory
src_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/data/'
dst_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/'
dst_dir_mask = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/ribbon_space-T2w_dseg'
dst_dir_imgs = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/img_classification/restore_T2w'

if not os.path.isdir(dst_directory):
    os.mkdir(dst_directory)

if not os.path.isdir(dst_dir_mask):
    os.mkdir(dst_dir_mask)

if not os.path.isdir(dst_dir_imgs):
    os.mkdir(dst_dir_imgs)

##### obtain MRI date

mri_directory = home + '/Desktop/UPC/DD/Final_Project/Data_analysis/data/'

# get directories for each subject
mri_subjects = glob.glob(mri_directory + 'sub-*' + '/sub-*_sessions.tsv', recursive=True)

mri_list = []

for i in range(len(mri_subjects)):
    with open(mri_subjects[i], newline='') as tsvfile:
        subject = mri_subjects[i].split('/')[9].split('-')[1]
        reader_mri = csv.reader(tsvfile, delimiter='\t')
        data_mri = list(reader_mri)
        data_mri.pop(0)

        for j in range(len(data_mri)):
            # print(str(j)+'\t'+str(data_mri))
            vect = [subject, data_mri[j][1]]
            mri_list.append(vect)


def take_second(elem):
    return elem[1]


mri_date_s = sorted(mri_list, key=take_second)

mri_date_20 = []
count_20_ = 0

for i in range(len(mri_date_s)):
    if float(mri_date_s[i][1]) < 40 and count_20_ < 20:
        if not any(mri_date_s[i][0] in s for s in mri_date_20):
            # print(mri_date_s[i])
            count_20_ += 1
            mri_date_20.append(mri_date_s[i])  # ["id", "birth_date"]

# print(mri_date_20)

##### obtain birth date

with open(src_directory + 'participants.tsv', newline='') as tsvfile:
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
birth_date_20 = []
count_20 = 0

for i in range(len(birth_date_s)):
    if float(birth_date_s[i][1]) < 30 and count_20 <= 50:
        count_20 += 1
        birth_date_20.append(birth_date_s[i])

# print(birth_date_20)

src_subjects = glob.glob(src_directory + 'sub-*', recursive=True)
num_cortex = 0

for subject in src_subjects:

    sub = subject.split('/')[9].split('-')[1]

    # select the subject that are in the list of the 20 youngest
    if any(sub in s for s in mri_date_20):  # birth_date_20

        src_sessions = glob.glob(subject + '/ses-*', recursive=True)

        for session in src_sessions:

            # from here the masks are obtained
            src_cortex = glob.glob(session + '/*-ribbon_space-T2w_dseg.nii.gz')
            if len(src_cortex) == 1 and num_cortex < 20:
                shutil.copy2(src_cortex[0], dst_dir_mask)

                # from here the images are used if the corresponding image can be obtained
                src_t2w = glob.glob(session + '/*-restore_T2w.nii.gz')[0]
                shutil.copy2(src_t2w, dst_dir_imgs)

                num_cortex += 1

img_t2w = glob.glob(dst_dir_imgs + '/*-restore_T2w.nii.gz', recursive=True)
img_cortex = glob.glob(dst_dir_mask + '/*-ribbon_space-T2w_dseg.nii.gz', recursive=True)

count = 0
num_imgs = 20

# indicate for which directory we want to generate masks
dirr = img_cortex

for j in range(len(dirr)):
    if count == num_imgs:
        break
    else:
        sub = dirr[j].split('/')[10].split('_')[0].split('-')[1]
        ses = dirr[j].split('/')[10].split('_')[1].split('-')[1]

        img_ = nib.load(dirr[j])
        data_m = img_.get_fdata()
        mask_m = np.random.rand(*data_m.shape)
        # id_ = uuid.uuid1()

        # lettersAndDigits = string.ascii_letters + string.digits
        # id_ = ''.join((random.choice(lettersAndDigits) for i in range(8)))

        for i in range(len(data_m[1, 1, :])):
            # values of the cortical plate
            mask_m[:, :, i] = (data_m[:, :, i] == 42)
            mask_m[:, :, i] += (data_m[:, :, i] == 3)

        for k in range(len(img_t2w)):
            sub_ = img_t2w[k].split('/')[10].split('_')[0].split('-')[1]

            if sub == sub_:

                ses_ = img_t2w[k].split('/')[10].split('_')[1].split('-')[1]

                if ses == ses_:
                    mask_ = mask_m.astype(np.uint16)
                    mask_n = nib.Nifti1Image(mask_, img_.affine, img_.header)

                    data_ = data_m.astype(np.uint16)
                    data_n = nib.Nifti1Image(data_, img_.affine, img_.header)

                    nib.save(data_n,
                             dst_directory + 'ribbon_space-T2w_dseg/imgs/' + str(sub) + '_' + str(ses) + '.nii.gz')
                    nib.save(mask_n,
                             dst_directory + 'ribbon_space-T2w_dseg/mask/' + str(sub) + '_' + str(ses) + '.nii.gz')

                    # copy the corresponding image of the mask with the same name
                    shutil.copy2(img_t2w[k], dst_directory + 'restore_T2w/' + str(sub) + '_' + str(ses) + '.nii.gz')

                    count += 1

                    print('Count: ' + str(count))
                    print('Subject: ' + str(sub) + ' \t Session: ' + str(ses))
