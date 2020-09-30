import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from os import listdir
from os.path import splitext
from glob import glob
import nibabel as nib

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, np.nanmin(full_img)))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask, img_ = False):
    mask_ = mask.astype(np.float32)
  
    if img_ == False:
        mask_n = nib.Nifti1Image(mask_, affine = None)
    else: 
        mask_n = nib.Nifti1Image(mask_, img_.affine, img_.header)
    
    return mask_n


def binarize_mask(mask, threshold = 0.5):
    mask_binarized = np.where(mask>threshold, 1, 0)
    return mask_binarized


if __name__ == "__main__":
    args = get_args()
    in_files = args.input[0]
    output_files = args.output[0]
 
    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    ids = [splitext(splitext(file)[0])[0] for file in listdir(in_files) if not file.startswith('.')]
    
    if 'GT' in ids: 
        ids.remove('GT')
    if 'GT_drawem9' in ids:
        ids.remove('GT_drawem9')

    for idx in ids:
        logging.info("\nPredicting image {} ...".format(idx))
        print('IDX: '+str(idx))
        img_file = glob(in_files + idx  + '.nii.gz')[0]
        
        img = nib.load(img_file).get_fdata()
        
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            mask_binarized = binarize_mask(mask)
            result = mask_to_image(mask_binarized)
            nib.save(result, output_files+idx+'.nii.gz')

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(idx))
            plot_img_and_mask(img, mask)
