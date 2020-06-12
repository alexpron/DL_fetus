import nibabel as nib
from os.path import expanduser
home = expanduser("~")
import os
import glob
import random
import shutil
import csv

mri_file = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dHCP/img2D/mri_date.csv'

with open(mri_file, newline='') as tsvfile:
    reader_mri = csv.reader(tsvfile, delimiter='\t')
    data_mri = list(reader_mri)

test_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dHCP/img2D/test_im2D/'
imgs_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dHCP/img2D/both/imgs/'
mask_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/dHCP/img2D/both/mask/'

if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

if not os.path.isdir(test_dir+'GT_/'):
    os.mkdir(test_dir+'GT_')

imgs_list = glob.glob(imgs_dir+'*_*.nii.gz')
mask_list = glob.glob(mask_dir+'*_*.nii.gz')

num = 20
count_im = 0
random_selection = True

# select num random images and the respective mask
if random_selection == True:
    while count_im < num:
        test_im = random.choice(imgs_list)
        test_id = test_im.split('/')[12]
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
                    shutil.copy2(mask_im, test_dir+'GT_/GT_test_'+str(count_im)+'_age_'+str(mri_date)+'.nii.gz')
                    # nib.save(data_n, src_both_dir+'/imgs/'+str(sub_i)+'_'+str(ses_i)+'_'+str(j)+'.nii.gz')
                    count_im += 1
# select all the images corresponding to the same sub and ses
else:
    test_im = random.choice(imgs_list)
    test_id = test_im.split('/')[12]
    sub = test_id.split('_')[0]
    ses = test_id.split('_')[1]

    print('test_id: ' + str(test_id))

    imgs_test_list = sorted(glob.glob(imgs_dir + str(sub) + '_' + str(ses) + '_*.nii.gz'))
    mask_test_list = sorted(glob.glob(mask_dir + str(sub) + '_' + str(ses) + '_*.nii.gz'))

    print('imgs list: ' + str(len(imgs_test_list)))

    for i in range(len(imgs_test_list)):
        test_im = imgs_test_list[i]
        mask_im = mask_test_list[i]
        print('IMAGE: ' + str(imgs_test_list[i].split('/')[12]))
        print('MASK: ' + str(mask_test_list[i].split('/')[12]))
        test_id = test_im.split('/')[12]
        idx = test_id.split('_')[2].split('.')[0]
        print('INDEX: ' + str(idx))
        for i in range(len(data_mri)):
            if data_mri[i][0] == sub:
                if data_mri[i][1] == ses:
                    print('')
                    mri_date = data_mri[i][2]
                    shutil.copy2(test_im,
                                 test_dir + 'test_' + str(sub) + '_' + str(ses) + '_' + str(idx) + '_age_' + str(
                                     mri_date) + '.nii.gz')
                    shutil.copy2(mask_im,
                                 test_dir + 'GT_/GT_test_' + str(sub) + '_' + str(ses) + '_' + str(idx) + '_age_' + str(
                                     mri_date) + '.nii.gz')
                    print(str(sub) + '_' + str(ses) + '_' + str(idx) + '_age_' + str(mri_date) + '.nii.gz')