# [CAVE: Cerebral artery-vein segmentation in digital subtraction angiography](https://doi.org/10.1016/j.compmedimag.2024.102392)
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
4. retrieve trained models
```bash
git lfs install
git lfs pull
```

## Description
This code implements two deep learning-based models for vessel and artery-vein segmentation in digital subtraction angiography.

It includes one baseline method that uses U-Net on minimum intensity projection (MinIP) images of DSA, and the proposed CAVE method, which is a spatio-temporal U-Net based model that uses complete DSA series with variable frame length as input. This is the first work in deep learning based artery-vein segmentation in DSA. Via comparisons with U-Net, we
demonstrate the added benefit of exploiting spatio-temporal characteristics of blood flow in artery-vein segmentation.

Both the code and trained models are shared in this repository. To fetch the trained models, you would need git lfs.


## Usage
**Note : Use Python 3.6 or newer**

### Training

```console
> python train.py -h
usage: train.py [-h] [--input-type INPUT_TYPE] [--label-type LABEL_TYPE] [--epochs E] [--batch-size B] [--accum-batches ACCUM_BATCHES] [--learning-rate LR] [--rnn RNN] [--rnn_kernel RNN_KERNEL] [--rnn_layers RNN_LAYERS] [--num_heads NUM_HEADS] [--load LOAD] [--img_scale IMG_SCALE] [--amp] [--exp_group EXP_GROUP]

Train the UNet on images and target masks

options:
  -h, --help            show this help message and exit
  --input-type INPUT_TYPE, -i INPUT_TYPE
                        Model input - minip or sequence.
  --label-type LABEL_TYPE, -t LABEL_TYPE
                        Label type - vessel (binary) or av (2 classes).
  --epochs E, -e E      Number of epochs.
  --batch-size B, -b B  Batch size.
  --accum-batches ACCUM_BATCHES, -a ACCUM_BATCHES
                        Gradient accumulation batches.
  --learning-rate LR, -l LR
                        Learning rate
  --rnn RNN, -r RNN     RNN type: convGRU, convLSTM, or TemporalTransformer.
  --rnn_kernel RNN_KERNEL, -k RNN_KERNEL
                        RNN kernel: 1 (1x1) or 3 (3x3).
  --rnn_layers RNN_LAYERS, -n RNN_LAYERS
                        Number of RNN layers.
  --num_heads NUM_HEADS
                        Number of transformer attention heads.
  --load LOAD, -f LOAD  Load model from a .pth file
  --img_scale IMG_SCALE, -s IMG_SCALE
                        Downscaling factor of the images
  --amp                 Use mixed precision.
  --exp_group EXP_GROUP, -g EXP_GROUP
                        Set wandb group name.

```

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training, your best model is saved to `checkpoints/checkpoint.pt`, as well as in wandb folder. 
You can test the model on other images using the following commands.

To segment a single image and save the results:

`python predict.py filepath/to/dsa_series.dcm filepath/to/output_segmentation.png filepath/to/checkpoint.pt`

To segment a set of DSA series and save the results:

`python predict.py dirpath/to/dsa_series dirpath/to/output_segmentations filepath/to/checkpoint.pt`

```console
> python predict.py -h
usage: predict.py [-h] [--input-type INPUT_TYPE] [--label-type LABEL_TYPE] [--rnn RNN] [--rnn_kernel RNN_KERNEL] [--rnn_layers RNN_LAYERS] [--img_size IMG_SIZE] [--amp] in_img_path out_img_path model

Train the UNet on images and target masks

positional arguments:
  in_img_path           Input image to be segmented.
  out_img_path          Segmentation result image.
  model                 Load model from a .pt file

options:
  -h, --help            show this help message and exit
  --input-type INPUT_TYPE, -i INPUT_TYPE
                        Model input - minip or sequence.
  --label-type LABEL_TYPE, -t LABEL_TYPE
                        Label type - vessel (binary) or av (4 classes).
  --rnn RNN, -r RNN     RNN type: convGRU or convLSTM.
  --rnn_kernel RNN_KERNEL, -k RNN_KERNEL
                        RNN kernel: 1 (1x1) or 3 (3x3).
  --rnn_layers RNN_LAYERS, -n RNN_LAYERS
                        Number of RNN layers.
  --img_size IMG_SIZE, -s IMG_SIZE
                        Targe image size for resizing images
  --amp                 Use mixed precision.
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
