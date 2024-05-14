import argparse
import glob
import logging
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import colors
from skimage.transform import resize
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Visualize DSA images and mask overlays')
    parser.add_argument('minip', type=str, help='Path of (dir to) input MinIP DSA image.')
    parser.add_argument('gt', help='Path of (dir to) binary vessel mask.')
    parser.add_argument('--pd_unet', type=str, default=False, help='Path of (dir to) predicted mask image of UNet')
    parser.add_argument('--pd_st_unet', type=str, default=False,
                        help='Path of (dir to) predicted mask image of ST UNet')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Image scale factor')
    parser.add_argument('--out', '-o', type=str, default=False, help='Path of (dir to) output visualization image.')

    return parser.parse_args()


if __name__ == '__main__':
    log_filepath = 'log/{}.log'.format(Path(__file__).stem)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler(sys.stdout)])

    '''Global settings'''
    args = get_args()
    pd_unet_imgs, pd_st_unet_imgs = [], []
    pd_unet_scores, pd_st_unet_scores = {}, {}

    minip_list = glob.glob(os.path.join(args.minip, '*.png')) if os.path.isdir(args.minip) else [args.minip]
    gt_list = glob.glob(os.path.join(args.gt, '*.png')) if os.path.isdir(args.gt) else [args.gt]
    minip_list = [m for m in minip_list if any(Path(m).name in s for s in gt_list)]

    methods_imgs = []
    methods_scores = []
    methods_title = []
    if args.pd_unet:
        pd_unet_imgs = glob.glob(os.path.join(args.pd_unet, '*.png')) if os.path.isdir(args.pd_unet) else [args.pd_unet]
        result_csv_dirpath = args.pd_unet if os.path.isdir(args.pd_unet) else Path(args.pd_unet).parent
        result_csv_path = os.path.join(result_csv_dirpath, 'results.csv')
        pd_unet_scores = pd.read_csv(result_csv_path, index_col='filename')
        minip_list = [m for m in minip_list if Path(m).stem in pd_unet_scores.index.values.tolist()]
        methods_imgs.append(pd_unet_imgs)
        methods_scores.append(pd_unet_scores)
        methods_title.append('U-Net')
    if args.pd_st_unet:
        pd_st_unet_imgs = glob.glob(os.path.join(args.pd_st_unet, '*.png')) if os.path.isdir(
            args.pd_st_unet) else [args.pd_st_unet]
        result_csv_dirpath = args.pd_st_unet if os.path.isdir(args.pd_st_unet) else Path(args.pd_st_unet).parent
        result_csv_path = os.path.join(result_csv_dirpath, 'results.csv')
        pd_st_unet_scores = pd.read_csv(result_csv_path, index_col='filename')
        minip_list = [m for m in minip_list if Path(m).stem in pd_st_unet_scores.index.values.tolist()]
        methods_imgs.append(pd_st_unet_imgs)
        methods_scores.append(pd_st_unet_scores)
        methods_title.append('ST U-Net')

    for minip_path in sorted(minip_list):
        filestem = Path(minip_path).stem
        filename = Path(minip_path).name
        gt = [m for m in gt_list if filename in m]
        if len(gt) != 1:
            logging.error("{} reference segmentation not found: {}".format(('No', 'Multiple')[len(gt) > 1], minip_path))
            continue
        gt = np.asarray(Image.open(gt[0]))
        minip = np.asarray(Image.open(minip_path))
        # resize to the same size of the model input/output
        ts = round(1024 * args.scale)
        gt = resize(gt, (ts, ts), order=0, anti_aliasing=False, preserve_range=True).astype(np.uint8)
        minip = resize(minip, (ts, ts), anti_aliasing=False, preserve_range=True)

        '''Prepare visualization'''
        ncols = 4 if args.pd_unet else 2
        ncols = ncols + 2 if args.pd_st_unet else ncols
        fig, ax = plt.subplots(nrows=4, ncols=ncols)
        fig.tight_layout()
        plt.rcParams.update({'axes.titlesize': 5})
        plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
        # preparing colormap
        cmap_av = colors.ListedColormap(['white', 'red', 'blue', 'purple'])
        cmap_artery = colors.ListedColormap(['white', 'red'])
        cmap_vein = colors.ListedColormap(['white', 'blue'])
        cmap_vessel = colors.ListedColormap(['white', 'green'])

        # col 1: minip
        icol = 0
        ax[0, icol].imshow(minip, cmap='gray', interpolation='none')
        ax[0, icol].axis('off'), ax[0, icol].set_title('MinIP')
        ax[1, icol].imshow(minip, cmap='gray', interpolation='none'), ax[1, icol].axis('off')
        ax[2, icol].imshow(minip, cmap='gray', interpolation='none'), ax[2, icol].axis('off')
        ax[3, icol].imshow(minip, cmap='gray', interpolation='none'), ax[3, icol].axis('off')
        # col 2: minip with ground truth overlay
        icol += 1
        '''artery'''
        gt_artery = np.logical_or(gt == 1, gt == 3)
        ax[0, icol].imshow(minip, cmap='gray', interpolation='none')
        ax[0, icol].axis('off'), ax[0, icol].set_title('Reference')
        ax[0, icol].imshow(gt_artery, cmap=cmap_artery, alpha=0.5, interpolation='none')
        '''vein'''
        gt_vein = np.logical_or(gt == 2, gt == 3)
        ax[1, icol].imshow(minip, cmap='gray', interpolation='none'), ax[1, icol].axis('off')
        ax[1, icol].imshow(gt_vein, cmap=cmap_vein, alpha=0.5, interpolation='none')
        '''artery and vein'''
        ax[2, icol].imshow(minip, cmap='gray', interpolation='none'), ax[2, icol].axis('off')
        ax[2, icol].imshow(gt, cmap=cmap_av, alpha=0.5, interpolation='none')
        '''vessel'''
        gt_vessel = gt > 0
        ax[3, icol].imshow(minip, cmap='gray', interpolation='none'), ax[3, icol].axis('off')
        ax[3, icol].imshow(gt_vessel, cmap=cmap_vessel, alpha=0.5, interpolation='none')

        # col 3 and 4
        for pd_imgs, pd_scores, method in zip(methods_imgs, methods_scores, methods_title):
            # read predicted mask
            pd = [m for m in pd_imgs if filename in m]
            score = pd_scores.loc[filestem]
            score_str = '{:.3f},a{:.3f},v{:.3f}'.format(score.dice, score.adice, score.vdice)
            if len(pd) != 1:
                logging.error("{} {} results found: {}".format(('No', 'Multiple')[len(pd) > 1], method, minip_path))
            else:
                pd = np.asarray(Image.open(pd[0]))
                icol += 1
                '''artery'''
                pd_artery = np.logical_or(pd == 1, pd == 3)
                ax[0, icol].imshow(minip, cmap='gray', interpolation='none')
                ax[0, icol].imshow(pd_artery, cmap=cmap_artery, alpha=0.5, interpolation='none')
                ax[0, icol].axis('off'), ax[0, icol].set_title('{}({})'.format(method, score_str))
                '''vein'''
                pd_vein = np.logical_or(pd == 2, pd == 3)
                ax[1, icol].imshow(minip, cmap='gray', interpolation='none')
                ax[1, icol].imshow(pd_vein, cmap=cmap_vein, alpha=0.5, interpolation='none'), ax[1, icol].axis('off')
                '''artery and vein'''
                ax[2, icol].imshow(minip, cmap='gray', interpolation='none')
                ax[2, icol].imshow(pd, cmap=cmap_av, alpha=0.5, interpolation='none'), ax[2, icol].axis('off')
                '''vessel'''
                pd_vessel = pd > 0
                ax[3, icol].imshow(minip, cmap='gray', interpolation='none')
                ax[3, icol].imshow(pd_vessel, cmap=cmap_vessel, alpha=0.5, interpolation='none')
                ax[3, icol].axis('off')

                icol += 1
                '''artery'''
                cmap_diff = colors.ListedColormap(['white', 'orangered', 'dodgerblue'])
                diff_artery = np.zeros_like(pd)
                diff_artery[(gt_artery == 0) & (pd_artery > 0)] = 1
                diff_artery[(gt_artery > 0) & (pd_artery == 0)] = 2
                ax[0, icol].imshow(diff_artery, cmap=cmap_diff, interpolation='none')
                ax[0, icol].axis('off'), ax[0, icol].set_title('Error')
                '''vein'''
                diff_vein = np.zeros_like(pd)
                diff_vein[(gt_vein == 0) & (pd_vein > 0)] = 1
                diff_vein[(gt_vein > 0) & (pd_vein == 0)] = 2
                ax[1, icol].imshow(diff_vein, cmap=cmap_diff, interpolation='none'), ax[1, icol].axis('off')
                '''artery and vein'''
                diff_av = np.zeros_like(pd)
                diff_av[(gt == 0) & (pd > 0)] = 1
                diff_av[(gt > 0) & (pd != gt) & (pd != 3)] = 2
                ax[2, icol].imshow(diff_av, cmap=cmap_diff, interpolation='none'), ax[2, icol].axis('off')
                '''vessel'''
                diff_vessel = np.zeros_like(pd)
                diff_vessel[(gt_vessel == 0) & (pd_vessel > 0)] = 1
                diff_vessel[(gt_vessel > 0) & (pd_vessel == 0)] = 2
                ax[3, icol].imshow(diff_vessel, cmap=cmap_diff, interpolation='none'), ax[3, icol].axis('off')

        if args.out:
            out_path = args.out if args.out.endswith('.png') else os.path.join(args.out, filename)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches='tight', dpi=1200)
        else:
            plt.show()
        plt.close()
