import nibabel as nib
from os.path import expanduser
home = expanduser("~")
import os
import glob
import random
import shutil
import csv

mri_file = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/mri_date.csv'

with open(mri_file, newline='') as tsvfile:
    reader_mri = csv.reader(tsvfile, delimiter='\t')
    data_mri = list(reader_mri)

test_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/test_im2D/'
imgs_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/both/imgs/'
mask_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/both/mask/'

imgs_list = glob.glob(imgs_dir+'*_*.nii.gz', recursive=True)
mask_list = glob.glob(mask_dir+'*_*.nii.gz', recursive=True)

num = 10
count_im = 0

while count_im < num:
    test_im = random.choice(imgs_list)
    test_id = test_im.split('/')[11]
    print('test_id: '+str(test_id))
    sub = test_id.split('_')[0]
    ses = test_id.split('_')[1]
    # find the date of the mri
    for i in range(len(data_mri)):
        if data_mri[i][0] == sub:
            if data_mri[i][1] == ses:
                mri_date = data_mri[i][2]
                # print('MRI date: '+str(mri_date))
                # return the index of the same subject and session as the image
                match = mask_list.index(mask_dir+test_id)
                # print('index: '+str(match))
                mask_im = mask_list[match]
                # print('mask_im: '+str(mask_im))
                # print(test_im)
                shutil.copy2(test_im, test_dir+'test_'+str(count_im)+'_age_'+str(mri_date)+'.nii.gz')
                shutil.copy2(test_im, test_dir+'GT_/GT_test_'+str(count_im)+'_age_'+str(mri_date)+'.nii.gz')
                # nib.save(data_n, src_both_dir+'/imgs/'+str(sub_i)+'_'+str(ses_i)+'_'+str(j)+'.nii.gz')
                count_im += 1