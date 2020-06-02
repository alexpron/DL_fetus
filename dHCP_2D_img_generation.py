import nibabel as nib
from os.path import expanduser
home = expanduser("~")
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import random

src_imgs_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/restore_T2w/'
src_mask_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/ribbon_space-T2w_dseg/mask/'
src_both_dir = home+'/Desktop/UPC/DD/Final_Project/Data_analysis/img2D/both/'

imgs_list = glob.glob(src_imgs_dir+'*_*.nii.gz', recursive=True)
mask_list = glob.glob(src_mask_dir+'*_*.nii.gz', recursive=True)

count = 0

for i in range(len(imgs_list)):

    # read the mask and the corresponding image
    img_ = nib.load(imgs_list[i])
    img = img_.get_fdata()
    mask_ = nib.load(mask_list[i])
    mask = mask_.get_fdata()

    # mask for the plot
    # mask_im = np.random.rand(*mask.shape)

    sub_i = imgs_list[i].split('/')[10].split('_')[0]
    sub_m = mask_list[i].split('/')[11].split('_')[0]

    # check we are working with the correct images
    if sub_i == sub_m:
        ses_i = imgs_list[i].split('/')[10].split('_')[1].split('.')[0]
        ses_m = mask_list[i].split('/')[11].split('_')[1].split('.')[0]

        if ses_i == ses_m:
            print(str(i)+': Working with subject ' + str(sub_i) + ' and session ' + str(ses_i))

            for j in range(len(img[1, 1, :])):
                # print('Next slice')
                # if there are non zero elements we save the image
                if np.count_nonzero(mask[:, :, j]) != 0:

                    # mask_im[mask == 0] = np.nan

                    # plt.figure()
                    # plt.imshow(img[:, :, j], cmap='gray')
                    # plt.imshow(mask_im[:,:,j], alpha=1)
                    # plt.savefig(src_both_dir + 'both_' + str(sub_i) + '_' + str(ses_i) + '_' + str(j) + '.png')
                    # plt.show()
                    # plt.close()

                    data_ = img[:, :, j].astype(np.uint8)
                    data_n = nib.Nifti1Image(data_, img_.affine, img_.header)

                    maskk_ = mask[:, :, j].astype(np.uint8)
                    mask_n = nib.Nifti1Image(maskk_, img_.affine, img_.header)

                    nib.save(data_n, src_both_dir + '/imgs/' + str(sub_i) + '_' + str(ses_i) + '_' + str(j) + '.nii.gz')
                    nib.save(mask_n, src_both_dir + '/mask/' + str(sub_i) + '_' + str(ses_i) + '_' + str(j) + '.nii.gz')

                    count += 1

# create a test_set of XXX images

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
    # return the index of the same subject and session as the image
    match = mask_list.index(mask_dir+test_id)
    # print('index: '+str(match))
    mask_im = mask_list[match]
    # print('mask_im: '+str(mask_im))
    # print(test_im)
    shutil.copy2(test_im, test_dir+'test_'+str(count_im)+'.nii.gz')
    shutil.copy2(mask_im, test_dir+'GT_/GT_test_'+str(count_im)+'.nii.gz')
    # nib.save(data_n, src_both_dir+'/imgs/'+str(sub_i)+'_'+str(ses_i)+'_'+str(j)+'.nii.gz')
    count_im += 1