import torch
import torch.nn.functional as F
from PIL import Image
import wandb
import torch.nn as nn
from utils.metrics import dice_coeff, dice_loss, acc_sens_spec
import os
import numpy as np
from pathlib import Path
import pandas as pd
import input_paths as I
import time

criterion = nn.CrossEntropyLoss()


def evaluate(net, dataloader, device=torch.device('cuda'), mode='unknown', wb=None, save=False):
    """
    Important Note: make sure the batch_size is set to one in the dataloader.
    Otherwise, the logging might be missing and the calculated values might be incorrect.
    """

    if len(dataloader) == 0:
        raise ValueError("Empty validation set.")

    net.eval()
    dices = []
    artery_dices = []
    vein_dices = []
    accuracies = []
    specificities = []
    sensitivities = []
    epoch_loss = 0
    results = []
    inf_times = []

    # iterate over the validation set
    for batch in dataloader:
        images, masks_true, filename = batch['image'], batch['mask'], batch['filename'][0]
        result = {'filename': filename}
        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        masks_true = masks_true.to(device=device, dtype=torch.long)

        # converting the true_masks from 4 classes to 2 classes
        if net.n_classes == 2:
            artery_masks = torch.logical_or(masks_true == 1, masks_true == 3)
            vein_masks = torch.logical_or(masks_true == 2, masks_true == 3)
            bg_masks = (masks_true == 0)
            masks_true = torch.stack((bg_masks, artery_masks, vein_masks), dim=1).float()
        if net.n_classes == 1:
            masks_true = F.one_hot(masks_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            start = time.time()
            masks_pred = torch.sigmoid(net(images))
            end = time.time()
            inf_times.append(end - start)
            loss = criterion(masks_pred, masks_true[:, 1:, ...]) + dice_loss(masks_pred, masks_true[:, 1:, ...],
                                                                              multiclass=True)
            epoch_loss += loss.item()

            # convert to one-hot format
            if net.n_classes == 1:
                masks_pred_one_hot = torch.squeeze(masks_pred, 1) >= 0.5
                masks_pred_one_hot = F.one_hot(masks_pred_one_hot.to(torch.int64), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dices.append(dice_coeff(masks_pred_one_hot[:, 1, ...], masks_true[:, 1, ...], reduce_batch_first=False, multiclass=True))
                if save:
                    mask_vis = Image.fromarray(masks_pred_one_hot[0].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
            else:
                masks_pred_one_hot = (masks_pred >= 0.5)
                bg_masks_pred_one_hot = torch.logical_not(
                    torch.logical_or(masks_pred_one_hot[:, 0, ...], masks_pred_one_hot[:, 1, ...]))
                masks_pred_one_hot = torch.stack(
                    (bg_masks_pred_one_hot, masks_pred_one_hot[:, 0, ...], masks_pred_one_hot[:, 1, ...]),
                    dim=1).float()
                # compute the Dice score, ignoring background
                dice = dice_coeff(masks_pred_one_hot[:, 1:, ...], masks_true[:, 1:, ...], reduce_batch_first=False,
                                  multiclass=True)
                dices.append(dice)
                adice = dice_coeff(masks_pred_one_hot[:, 1, ...], masks_true[:, 1, ...], reduce_batch_first=False)
                artery_dices.append(adice)
                vdice = dice_coeff(masks_pred_one_hot[:, 2, ...], masks_true[:, 2, ...], reduce_batch_first=False)
                vein_dices.append(vdice)
                if save:
                    masks_av = torch.logical_or(masks_pred_one_hot[:, 1, ...], masks_pred_one_hot[:, 2, ...])
                    masks_artery = torch.logical_and(masks_pred_one_hot[:, 1, ...],
                                                     torch.logical_not(masks_pred_one_hot[:, 2, ...]))
                    masks_vein = torch.logical_and(torch.logical_not(masks_pred_one_hot[:, 1, ...]),
                                                   masks_pred_one_hot[:, 2, ...])
                    mask_vis = torch.stack((bg_masks_pred_one_hot, masks_artery, masks_vein, masks_av), dim=1)
                    mask_vis = Image.fromarray(mask_vis[0].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
                    result.update({'dice': dice.item(), 'adice': adice.item(), 'vdice': vdice.item()})

            # compute the accuracy, including background
            acc = acc_sens_spec(masks_pred_one_hot, masks_true, reduce_batch_first=False, multiclass=True)
            accuracies.append(acc)
            # compute the sensitivity, excluding background
            sens = acc_sens_spec(masks_pred_one_hot[:, 1:, ...], masks_true[:, 1:, ...], reduce_batch_first=False,
                                 multiclass=True)
            sensitivities.append(sens)
            # compute the specificity, using only background
            spec = acc_sens_spec(masks_pred_one_hot[:, 0, ...], masks_true[:, 0, ...], reduce_batch_first=False,
                                 multiclass=True)
            specificities.append(spec)
            if save:
                result.update({'acc': acc.item(), 'sens': sens.item(), 'spec': spec.item()})
                results.append(result)
                save_path = os.path.join(I.result_dir, wb.name, '{}.png'.format(filename))
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                mask_vis.save(save_path)

            # if wb is not None:
            #     wb.log({
            #         '{}/true_{}'.format(mode, filename): wandb.Image(masks_true[0].float().cpu()),
            #         '{}/pred_{}'.format(mode, filename):
            #             wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu())})
            #     if images[0].dim() == 3:  # C*H*W
            #         wb.log({'{}/minip_{}'.format(mode, filename): wandb.Image(images[0].cpu())})
            #     else:  # T*C*H*W
            #         wb.log({
            #             '{}/sequence_{}'.format(mode, filename): wandb.Image(images[0].cpu()),
            #             '{}/minip_{}'.format(mode, filename): wandb.Image(torch.min(images[0], dim=0).values.cpu())})
    net.train()

    if save:
        df_result = pd.DataFrame.from_records(results)
        Path(I.result_dir).mkdir(parents=True, exist_ok=True)
        df_result.to_csv(os.path.join(I.result_dir, wb.name, 'results.csv'),
                         index=False, header=True, float_format='%.3f')

    dices = torch.stack(dices)
    dice_mean, dice_std = torch.mean(dices).item(), torch.std(dices).item()
    accuracies = torch.stack(accuracies)
    acc_mean, acc_std = torch.mean(accuracies).item(), torch.std(accuracies).item()
    specificities = torch.stack(specificities)
    spec_mean, spec_std = torch.mean(specificities).item(), torch.std(specificities).item()
    sensitivities = torch.stack(sensitivities)
    sens_mean, sens_std = torch.mean(sensitivities).item(), torch.std(sensitivities).item()

    summary_dict = {f'{mode}_dice_mean': dice_mean, '{}_dice_std'.format(mode): dice_std,
                    f'{mode}_acc_mean': acc_mean, '{}_acc_std'.format(mode): acc_std,
                    f'{mode}_spec_mean': spec_mean, '{}_spec_std'.format(mode): spec_std,
                    f'{mode}_sens_mean': sens_mean, '{}_sens_std'.format(mode): sens_std,
                    f'{mode}_inf_time_mean': np.mean(np.array(inf_times)),
                    f'{mode}_inf_time_std': np.std(np.array(inf_times))}
    if net.n_classes == 4:
        artery_dices, vein_dices = torch.stack(artery_dices), torch.stack(vein_dices)
        summary_dict['{}_adice_mean'.format(mode)] = torch.mean(artery_dices).item()
        summary_dict['{}_adice_std'.format(mode)] = torch.std(artery_dices).item()
        summary_dict['{}_vdice_mean'.format(mode)] = torch.mean(vein_dices).item()
        summary_dict['{}_vdice_std'.format(mode)] = torch.std(vein_dices).item()
    if wb is not None:
        wb.log(summary_dict)
    return dice_mean, epoch_loss / len(dataloader), summary_dict
