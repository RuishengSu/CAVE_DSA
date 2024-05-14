import argparse
import glob
import logging
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib import colors


def get_args():
    parser = argparse.ArgumentParser(description='Visualize DSA images and mask overlays')
    parser.add_argument('img', type=str, help='Path of (dir to) input MinIP DSA image.')
    parser.add_argument('--overlay', '-l', type=str, default=False, help='Type of overlay: [vessel, artery, vein, av].')
    parser.add_argument('--mask', '-m', type=str, default=False, help='Path of (dir to) binary vessel mask.')
    parser.add_argument('--out', '-o', type=str, default=False, help='Path of (dir to) output visualization image.')

    return parser.parse_args()


if __name__ == '__main__':
    log_filepath = 'log/{}.log'.format(Path(__file__).stem)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler(sys.stdout)])

    '''Global settings'''
    args = get_args()
    mask_list = []

    minip_list = glob.glob(os.path.join(args.img, '*.png')) if os.path.isdir(args.img) else [args.img]
    if args.overlay and args.mask:
        mask_list = glob.glob(os.path.join(args.mask, '*.png')) if os.path.isdir(args.mask) else [args.mask]

    for img_path in sorted(minip_list):
        if not (args.overlay and args.mask):
            continue
        assert args.overlay in ['vessel', 'artery', 'vein', 'av'], "Invalid overlay type!"

        vis = plt.imread(img_path)
        filename = Path(img_path).name
        mask = [m for m in mask_list if filename in m]
        if len(mask) != 1:
            logging.error("{} corresponding mask not found: {}".format(('No', 'Multiple')[len(mask)], img_path))
            continue
        mask = plt.imread(mask[0])

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(vis, cmap='gray', interpolation='none'), ax[0].axis('off'), ax[0].set_title('MinIP')
        if args.overlay == 'av':
            cmap = colors.ListedColormap(['white', 'red', 'blue', 'purple'])
        elif args.overlay == 'artery':
            cmap = colors.ListedColormap(['white', 'red'])
        elif args.overlay == 'vein':
            cmap = colors.ListedColormap(['white', 'blue'])
        else:
            cmap = colors.ListedColormap(['white', 'green'])
        ax[1].imshow(mask, cmap=cmap, interpolation='none'), ax[1].axis('off'), ax[1].set_title('Mask')
        ax[2].imshow(vis, cmap='gray', interpolation='none'), ax[2].axis('off'), ax[2].set_title('Overlay')
        ax[2].imshow(mask, cmap=cmap, alpha=0.5, interpolation='none')
        if args.out:
            out_path = args.out if args.out.endswith('.png') else os.path.join(args.out, filename)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches='tight', dpi=1200)
        else:
            plt.show()
        plt.close()
