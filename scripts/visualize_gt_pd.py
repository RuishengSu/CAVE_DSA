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
    parser.add_argument('overlay', type=str, help='Type of overlay: [vessel, artery, vein, av].')
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
    if args.pd_unet:
        pd_unet_imgs = glob.glob(os.path.join(args.pd_unet, '*.png')) if os.path.isdir(args.pd_unet) else [args.pd_unet]
        result_csv_dirpath = args.pd_unet if os.path.isdir(args.pd_unet) else Path(args.pd_unet).parent
        result_csv_path = os.path.join(result_csv_dirpath, 'results.csv')
        pd_unet_scores = pd.read_csv(result_csv_path, index_col='filename')
        minip_list = [m for m in minip_list if Path(m).stem in pd_unet_scores.index.values.tolist()]
    if args.pd_st_unet:
        pd_st_unet_imgs = glob.glob(os.path.join(args.pd_st_unet, '*.png')) if os.path.isdir(
            args.pd_st_unet) else [args.pd_st_unet]
        result_csv_dirpath = args.pd_st_unet if os.path.isdir(args.pd_st_unet) else Path(args.pd_st_unet).parent
        result_csv_path = os.path.join(result_csv_dirpath, 'results.csv')
        pd_st_unet_scores = pd.read_csv(result_csv_path, index_col='filename')
        minip_list = [m for m in minip_list if Path(m).stem in pd_st_unet_scores.index.values.tolist()]

    for minip_path in sorted(minip_list):
        assert args.overlay in ['vessel', 'artery', 'vein', 'av'], "Invalid overlay type!"

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
        fig, ax = plt.subplots(nrows=1, ncols=ncols)
        plt.rcParams.update({'axes.titlesize': 5})
        plt.subplots_adjust(wspace=0, hspace=0)
        # preparing colormap
        if args.overlay == 'av':
            cmap = colors.ListedColormap(['white', 'red', 'blue', 'purple'])
        elif args.overlay == 'artery':
            cmap = colors.ListedColormap(['white', 'red'])
        elif args.overlay == 'vein':
            cmap = colors.ListedColormap(['white', 'blue'])
        else:
            cmap = colors.ListedColormap(['white', 'green'])

        # col 1: minip
        icol = 0
        ax[icol].imshow(minip, cmap='gray', interpolation='none'), ax[icol].axis('off'), ax[icol].set_title('MinIP')
        # col 2: minip with ground truth overlay
        icol += 1
        ax[icol].imshow(minip, cmap='gray', interpolation='none'), ax[icol].axis('off'), ax[icol].set_title('Reference')
        ax[icol].imshow(gt, cmap=cmap, alpha=0.5, interpolation='none')
        # col 3 and 4
        if args.pd_unet:
            # read predicted mask
            pd = [m for m in pd_unet_imgs if filename in m]
            score = pd_unet_scores.loc[filestem]
            score_str = '{:.3f}'.format(score.dice)
            if ('adice' in score) and (args.overlay in ['av', 'artery']):
                score_str = score_str + ',a{:.3f}'.format(score.adice)
            if ('vdice' in score) and (args.overlay in ['av', 'vein']):
                score_str = score_str + ',v{:.3f}'.format(score.vdice)
            if len(pd) != 1:
                logging.error("{} U-Net results found: {}".format(('No', 'Multiple')[len(pd) > 1], minip_path))
            else:
                pd = np.asarray(Image.open(pd[0]))
                if args.overlay == 'artery':
                    pd[pd == 3] = 1
                    pd[pd != 1] = 0
                elif args.overlay == 'vein':
                    pd[pd == 3] = 2
                    pd[pd != 2] = 0
                    pd[pd == 2] = 1
                icol += 1
                ax[icol].imshow(minip, cmap='gray', interpolation='none')
                ax[icol].imshow(pd, cmap=cmap, alpha=0.5, interpolation='none')
                ax[icol].axis('off'), ax[icol].set_title('U-Net({})'.format(score_str))

                icol += 1
                gt_pd_diff = np.zeros_like(pd)
                gt_pd_diff[(gt == 0) & (pd > 0)] = 1
                gt_pd_diff[(gt > 0) & (gt != pd) & (pd != 3)] = 2
                ax[icol].imshow(gt_pd_diff, cmap=colors.ListedColormap(['white', 'orangered', 'dodgerblue']),
                                interpolation='none')
                ax[icol].axis('off'), ax[icol].set_title('Error')

        # col 5 and 6
        if args.pd_st_unet:
            # read predicted mask
            pd = [m for m in pd_st_unet_imgs if filename in m]
            score = pd_st_unet_scores.loc[filestem]
            score_str = '{:.3f}'.format(score.dice)
            if ('adice' in score) and (args.overlay in ['av', 'artery']):
                score_str = score_str + ',a{:.3f}'.format(score.adice)
            if ('vdice' in score) and (args.overlay in ['av', 'vein']):
                score_str = score_str + ',v{:.3f}'.format(score.vdice)
            if len(pd) != 1:
                logging.error(
                    "{} spatio-temporal U-Net results found: {}".format(('No', 'Multiple')[len(pd) > 1], minip_path))
            else:
                pd = np.asarray(Image.open(pd[0]))
                if args.overlay == 'artery':
                    pd[pd == 3] = 1
                    pd[pd != 1] = 0
                elif args.overlay == 'vein':
                    pd[pd == 3] = 2
                    pd[pd != 2] = 0
                    pd[pd == 2] = 1
                icol += 1
                ax[icol].imshow(minip, cmap='gray', interpolation='none')
                ax[icol].imshow(pd, cmap=cmap, alpha=0.5, interpolation='none')
                ax[icol].axis('off'), ax[icol].set_title('ST U-Net({})'.format(score_str))

                icol += 1
                gt_pd_diff = np.zeros_like(gt)
                gt_pd_diff[(gt == 0) & (pd > 0)] = 1
                gt_pd_diff[(gt > 0) & (gt != pd) & (pd != 3)] = 2
                ax[icol].imshow(gt_pd_diff, cmap=colors.ListedColormap(['white', 'orangered', 'dodgerblue']),
                                interpolation='none')
                ax[icol].axis('off'), ax[icol].set_title('Error')

        if args.out:
            out_path = args.out if args.out.endswith('.png') else os.path.join(args.out, filename)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches='tight', dpi=1200)
        else:
            plt.show()
        plt.close()
