import argparse
import logging
import os
from pathlib import Path
import sys
import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import input_paths
import wandb
from evaluate import evaluate
from unet import UNet, TemporalUNet, ConvLSTM, ConvGRU, TemporalTransformerUNet
from utils.data_loading import DSADataset
from utils.early_stopping import EarlyStopping
from utils.metrics import dice_loss


# TODO: make optimizer an arg
def train_net(net,
              epochs: int = 5,
              learning_rate: float = 0.001,
              accum_batches: int = 1,
              amp: bool = False,
              device=torch.device('cuda'),
              wandb_logging=None,
              output_checkpoint=None):
    assert output_checkpoint is not None, "Output checkpoint path must be set."
    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr=1e-8,
                                  verbose=True)  # goal: maximize Dice
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=50, verbose=True, mode='max')
    best_val_dice = 0

    '''Begin training'''
    for epoch in range(epochs):

        ###################
        # train the model #
        ###################
        net.train()
        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                images = batch['image']
                masks_true = batch['mask']

                assert images.shape[-3] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                masks_true = masks_true.to(device=device, dtype=torch.long)

                # converting the true_masks from 4 classes to 2 classes
                if net.n_classes == 2:
                    artery_masks = torch.logical_or(masks_true == 1, masks_true == 3)
                    vein_masks = torch.logical_or(masks_true == 2, masks_true == 3)
                    masks_true = torch.stack((artery_masks, vein_masks), dim=1).float()
                if net.n_classes == 1:
                    masks_true = torch.unsqueeze(masks_true, dim=1).float()

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, masks_true) \
                        + dice_loss(torch.sigmoid(masks_pred).float(), masks_true, multiclass=True)
                batches = (accum_batches, len(train_loader) % accum_batches)[
                    i + 1 > len(train_loader) - len(train_loader) % accum_batches]
                grad_scaler.scale(loss / batches).backward()
                if ((i + 1) % accum_batches == 0) or ((i + 1) == len(train_loader)):
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                pbar.update(images.shape[0])
                global_step += 1
                pbar.set_postfix(**{'train loss (batch)': loss.item()})

                wandb_logging.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

            ######################
            # validate the model #
            ######################
            val_dice, val_loss, _ = evaluate(net, val_loader, device, mode='val')
            best_val_dice = max(best_val_dice, val_dice)

            wandb_logging.log({
                'init learning rate': "{:.1e}".format(args.lr),
                'validation dice': val_dice,
                'validation dice (latest)': val_dice,
                'validation dice (best)': best_val_dice,
                # 'images': wandb.Image(images[0].cpu()),
                # 'masks': {
                #     'true': wandb.Image(true_masks[0].float().cpu()),
                #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                # },
                'step': global_step,
                'epoch': epoch,
            })

            # early_stopping needs the validation loss to check if it has decreased
            pbar_postfix_str = 'train loss (batch)={:.8f}, val loss (epoch)={:.8f}, ' \
                               'val dice (epoch)={:.8f}.'.format(loss, val_loss, val_dice)
            early_stopping(val_dice)  # this function does not save the best model.
            if early_stopping.save_model:
                torch.save(net.state_dict(), output_checkpoint)
                pbar_postfix_str += " Model saved."
            if early_stopping.counter != 0:
                pbar_postfix_str += ' EarlyStopping counter: {}/{}'.format(early_stopping.counter,
                                                                           early_stopping.patience)
            pbar.set_postfix_str(pbar_postfix_str)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break
            scheduler.step(val_dice)
    # load the checkpoint of the best model
    net.load_state_dict(torch.load(output_checkpoint))
    torch.save(net.state_dict(), os.path.join(wandb_logging.dir, 'checkpoint.pt'))


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--input-type', '-i', default='minip', help='Model input - minip or sequence.')
    parser.add_argument('--label-type', '-t', default='vessel', help='Label type - vessel (binary) or av (2 classes).')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size.')
    parser.add_argument('--accum-batches', '-a', type=int, default=1, help='Gradient accumulation batches.')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--rnn', '-r', type=str, default='ConvGRU', help='RNN type: convGRU, convLSTM, or TemporalTransformer.')
    parser.add_argument('--rnn_kernel', '-k', type=int, default=1, help='RNN kernel: 1 (1x1) or 3 (3x3).')
    parser.add_argument('--rnn_layers', '-n', type=int, default=2, help='Number of RNN layers.')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of transformer attention heads.')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--img_scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision.')
    parser.add_argument('--exp_group', '-g', type=str, default=None, help='Set wandb group name.')

    return parser.parse_args()


if __name__ == '__main__':
    log_filepath = 'log/{}.log'.format(Path(__file__).stem)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler(sys.stdout)])

    '''Global settings'''
    args = get_args()
    assert args.input_type in ['minip', 'sequence'], "Invalid input image type"
    assert args.label_type in ['vessel', 'av'], "Invalid label type"
    train_img_dir = os.path.join(input_paths.model_input_dir, 'train', 'imgs_{}'.format(args.input_type))
    train_mask_dir = os.path.join(input_paths.model_input_dir, 'train', 'masks_{}'.format(args.label_type))
    val_img_dir = os.path.join(input_paths.model_input_dir, 'val', 'imgs_{}'.format(args.input_type))
    val_mask_dir = os.path.join(input_paths.model_input_dir, 'val', 'masks_{}'.format(args.label_type))
    test_img_dir = os.path.join(input_paths.model_input_dir, 'test', 'imgs_{}'.format(args.input_type))
    test_mask_dir = os.path.join(input_paths.model_input_dir, 'test', 'masks_{}'.format(args.label_type))
    n_classes = (1, 2)[args.label_type == 'av']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    '''Set up the network'''
    if args.input_type == 'minip':
        net = UNet(n_channels=1, n_classes=n_classes, bilinear=True)
    else:
        if args.rnn in ['ConvLSTM', 'ConvGRU']:
            rnn = (ConvGRU, ConvLSTM)[args.rnn == 'ConvLSTM']
            kernel_size = (args.rnn_kernel, args.rnn_kernel)
            net = TemporalUNet(rnn, n_channels=1, kernel_size=kernel_size, num_layers=args.rnn_layers,
                               n_classes=n_classes, bilinear=True)
        elif args.rnn == "TemporalTransformer":
            net = TemporalTransformerUNet(n_channels=1, n_classes=n_classes, 
                                        H=int(1024*args.img_scale), W=int(1024*args.img_scale), bilinear=True,
                                          num_layers=args.rnn_layers, num_heads=args.num_heads)
        else:
            raise ValueError("Unsupported rnn type!")

    net.to(device=device)

    '''Whether to resume from an existing trained model.'''
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    '''1. Create dataset'''
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5),
    ])
    train_set = DSADataset(train_img_dir, train_mask_dir, scale=args.img_scale, transform=transform)
    val_set = DSADataset(val_img_dir, val_mask_dir, scale=args.img_scale)
    test_set = DSADataset(test_img_dir, test_mask_dir, scale=args.img_scale)

    '''2. Create data loaders'''
    loader_args = dict(num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size=1, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size=1, **loader_args)

    '''3. Set up Wandb'''
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', group=args.exp_group)
    experiment.config.update(args)
    experiment.define_metric("validation dice", summary="max")
    experiment.define_metric("train loss", summary="min")

    '''4. Train or load a model'''
    if not args.load:
        # make sure wandb is set which generates a random experiment name, which is used as a folder name.
        ckpt = os.path.join(input_paths.result_dir, experiment.name, 'checkpoint.pt')
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        train_net(net=net, epochs=args.epochs, accum_batches=args.accum_batches, learning_rate=args.lr,
                  amp=args.amp, device=device, wandb_logging=experiment, output_checkpoint=ckpt)
    else:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    '''5. Begin validation'''
    _, _, val_result = evaluate(net, val_loader, mode='val', device=device, wb=experiment)
    logging.info("Results (val ) ---- {}".format(val_result))

    '''6. Begin testing'''
    _, _, test_result = evaluate(net, test_loader, mode='test', device=device, wb=experiment, save=True)
    logging.info("Results (test) ---- {}".format(test_result))

    '''7. Write results to CSV files'''
    result = {'wandb': experiment.name}
    result.update(vars(args))
    result.update(val_result)
    result.update(test_result)
    df_result = pd.DataFrame.from_records([result])
    Path(input_paths.result_dir).mkdir(parents=True, exist_ok=True)
    df_result.to_csv(os.path.join(input_paths.result_dir, experiment.name, 'summary.csv'),
                     index=False, header=True, float_format='%.3f')
    logging.info("Done!")
