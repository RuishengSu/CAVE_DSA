# CAVE: Cerebral artery-vein segmentation in digital subtraction angiography
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

[//]: # (![input and output for a random image in the test dataset]&#40;https://i.imgur.com/GD8FcB7.png&#41;)


- [Quick start](#quick-start)
- [Description](#description)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)


## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
conda env create -n {env_name}
conda activate {env_name}
pip install -r requirements.txt
```

## Description
This code implements two deep learning-based models for vessel and artery-vein segmentation in digital subtraction angiography.
It includes one baseline method that uses U-Net on minimum intensity projection (MinIP) images of DSA, and the proposed CAVE method, which is a spatio-temporal U-Net based model that uses complete DSA series with variable frame length as input. This is the first work in deep learning based artery-vein segmentation in DSA. Via comparisons with U-Net, we
demonstrate the added benefit of exploiting spatio-temporal characteristics of blood flow in artery-vein segmentation.


## Usage
**Note : Use Python 3.6 or newer**

### Training

```console
> python train.py -h
usage: train.py [-h] [--input-type INPUT_TYPE] [--label-type LABEL_TYPE]
                [--epochs E] [--batch-size B] [--accum-batches ACCUM_BATCHES]
                [--learning-rate LR] [--rnn RNN] [--rnn_kernel RNN_KERNEL]
                [--load LOAD] [--img_scale IMG_SCALE] [--val-ratio VAL_RATIO]
                [--amp]

Baseline U-Net example: ('-i minip' will trigger the use of u-net)
python train.py -i minip -t av --amp -b 1 -a 1 -l 0.00001 -s 0.5

CAVE example: ('-i sequence' will trigger the usage of spatio-temporal u-net)
python train.py -i sequence -t av --amp -b 1 -a 1 -l 0.00001 -s 0.5

Train, validate and test the baseline UNet or CAVE U-Net.
optional arguments:
  -h, --help            show this help message and exit
  --input-type INPUT_TYPE, -i INPUT_TYPE
                        Model input - minip or sequence.
  --label-type LABEL_TYPE, -t LABEL_TYPE
                        Label type - vessel (binary) or av (4 classes).
  --epochs E, -e E      Number of epochs.
  --batch-size B, -b B  Batch size.
  --accum-batches ACCUM_BATCHES, -a ACCUM_BATCHES
                        Gradient accumulation batches.
  --learning-rate LR, -l LR
                        Learning rate
  --rnn RNN, -r RNN     RNN type: convGRU or convLSTM.
  --rnn_kernel RNN_KERNEL, -k RNN_KERNEL
                        RNN kernel: 1 (1x1) or 3 (3x3).
  --load LOAD, -f LOAD  Load model from a .pth file
  --img_scale IMG_SCALE, -s IMG_SCALE
                        Downscaling factor of the images
  --val-ratio VAL_RATIO, -v VAL_RATIO
                        Portion of the data that is used as validation (0-1)
  --amp                 Use mixed precision

```

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training, your best model is saved to `checkpoints/checkpoint.pt`, as well as in wandb folder. 
You can test the model on other images using the following commands.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg --model path/to/checkpoint.pt`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pt`.

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  
Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. 
Click on it to go to your dashboard. 
If you have an existing W&B account, you can link it by setting the `WANDB_API_KEY` environment variable.

---

Welcome to cite our paper if you find this useful!

[Su et al. CAVE: Cerebral artery-vein segmentation in digital subtraction angiography](https://doi.org/10.1016/j.compmedimag.2024.102392)
