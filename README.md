

# Scripts

`masks_generation_dHCP.ipynb` generates the 3D masks for the selected images.

`Normalisation.ipynb` normalises the 3D images. It also generates 2D images from the 3D stack images (for both training and testing datasets).

`UNetPytorch_vs_UNetDou.ipynb` compares the results obtained from both basic UNet and residual UNet (backbone network from the FetalCPSeg architecture).

`Res_UNet_Multiclass.ipynb` compares the results obtained from single label studies (CSF+V, CP, WM) with the results obtained from the multilabel segmentation of these three masks.


# Mesocentre

For setting up the environment in the Mesocentre:

Firs, start a session in the corresponding partition:

```shell script
srun -p partition -A user_id -t time --gres=gpu:1 --pty bash

partition: volta / pascal
```

Second, load the necessary modules:

```shell script
> module purge
> module load userspace/all
> module load python3/3.6.3
> module load cuda/10.1
```

Third, activate the virtual environment created in the folder of the code:

```shell script
> source .venv/bin/activate
```

## UNet: semantic segmentation with PyTorch

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 40)
  -b [B], --batch-size [B]
                        Batch size (default: 13)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 10.0)

```

The directories used for the training images and groundtruths are indicated in the begining of the python script `train.py` as well as the directory for the checkpoints. The name of the file that saves the dice and coefficient values is indicated as well in the script `train.py`.

Once the training has finished, the file that saves the dice coefficient values has to be plotted in order to find in which epoch the optimal result is achieved and, then, take this checkpoint as the `MODEL.pth`.

After selecting the `MODEL.pth` used for the predicitons, the script `predict.py` can be launched.

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```

When computing the training script, it is really important to create in advance the output directory. This will have to be indicated in the terminal as well as the input directory (where the test images are).

The data augmentation for this network is found in the `utils/dataset.py` script. The functions used are: `noise_injection()` and `get_augmentation()`.

For the multiclass labeling, the only modifications that have to be made are:

- change on the `train.py` file the number of classes. It should be `k` classes +1 (this extra class will correspond to the background).
- if working with an imbalanced dataset, modify the weights in the part where the criterion loss is declared.
- change on the `predict.py` file the number of classes as well. It will output an image with `k+1` chaannels (`k` classes + the background).

## 2D FetalCPSeg: semantic segmentation of cortical plate (CP)

In this code, all the parameters are found within the scripts `Train.py` and `Infer.py`, so they have to be tuned in advance.

For the `Train.py` execution, the different parameters that can be tuned in the script are:

```shell script
self.lr = 1e-3
self.weight_decay = 1e-4
self.batch_size = 32

self.num_iteration = 60000
self.val_fre = 200
self.pre_fre = 20

self.patch_size = 64

self.data_path = '../Data/imgs_2D/'
self.mask_path = '../Data/mask_2D/'
self.output_path = 'output_checkpoints/'
```

The `data_path` and `mask_path` are the path where the training images and groundtruths are stored. The `output_path` is where the checkpoints are saved. In this architecture there is no need for selecting the best checkpoint, the code does it automatically saving it as `best_val.pth.gz`.

For the `Infer.py` script, the parameter that need to be indicated is the one named `output_path` in the `Train.py` script as well as the directory where the test images are obtained from. Everything else is saved in there and it is done automatically.

The data augmentation for this network is found in the `DataOp.py` script. The functions used are: `noise_injection()` and `get_augmentation()`. These are inside the class `TrainGenerator.py`.

For the modifications and uses for this project the module `volumentations` has not been used.


