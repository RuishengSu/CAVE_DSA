import glob
import logging
import os
import random
import sys
from pathlib import Path

import cv2 as cv
import imageio
import nibabel as nib
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.interpolate import interp1d

import dicom_reader.reader as dicom_reader
import input_paths

logger = logging.getLogger(__name__)


def normalize(img):
    """Outputs image of type unsigned int"""
    image_minip_norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    return image_minip_norm.astype(np.uint8)


def cut_seq(seq, max_len):
    if seq.shape[0] > max_len:
        if np.sum(seq[0, ...]) >= np.sum(seq[-1, ...]):
            seq = seq[1:]
        else:
            seq = seq[:-1]
        seq = cut_seq(seq, max_len=max_len)
    return seq


def prepare_sequence_and_minip(row, mode):  # mode = 'train', 'val', or 'test'
    """Preparing nifti and minip for model input."""

    '''1. Converting raw dicom to nifti sequences with fixed 1 fps'''
    patient_id = row.patient_id
    dst_nii_path = os.path.join(input_paths.prepared_data_out_dir, mode, 'imgs_sequence',
                                '{}_{}'.format(patient_id, "{}.nii".format(row.filename)))
    Path(dst_nii_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Convert DSA to 1 fps and save to {}".format(dst_nii_path))

    src_dcm = os.path.join(input_paths.raw_dicom_dir, patient_id, "{}.dcm".format(row.filename))
    ds = dicom_reader.read_file(src_dcm, defer_size="1 KB", stop_before_pixels=False, force=True)
    if 'FrameTimeVector' in ds:
        if len(ds.FrameTimeVector) != ds.NumberOfFrames:
            logger.warning("Number of Frames ({}) does not match frame time vector length ({}): {}"
                           "".format(ds.NumberOfFrames, len(ds.FrameTimeVector), ds.FrameTimeVector))
            ds.FrameTimeVector = ds.FrameTimeVector[:ds.NumberOfFrames]
        cum_time_vector = np.cumsum(ds.FrameTimeVector)
    elif 'FrameTime' in ds:
        cum_time_vector = int(ds.FrameTime) * np.array(range(ds.NumberOfFrames))
    else:
        logger.error("Missing time info: {}".format(src_dcm))
        return
    non_duplicated_frame_indices = np.where(~pd.DataFrame(cum_time_vector).duplicated())
    cum_time_vector = cum_time_vector[non_duplicated_frame_indices]
    seq = ds.pixel_array[non_duplicated_frame_indices]
    # cum_time_vector = [e for i, e in enumerate(cum_time_vector) if i not in duplicated_frame_indices]
    # remove the first frame as it is most likely a non-contrast frame or an un-subtracted frame
    cum_time_vector, seq = cum_time_vector[1:], seq[1:]

    desired_frame_interval = 1000  # ms
    interp = interp1d(cum_time_vector, seq, axis=0)
    seq = interp(np.arange(cum_time_vector[0], cum_time_vector[-1], desired_frame_interval))

    MAX_LEN = 20  # Shorten unnecessarily long sequences.
    if seq.shape[0] > MAX_LEN:
        logger.warning("Sequence is unnecessarily long, "
                       "cutting it to {} frames based on minimum contrast.".format(MAX_LEN))
    seq = cut_seq(seq, max_len=MAX_LEN)

    seq = normalize(seq)
    ds.NumberOfFrames = seq.shape[0]
    ds.FrameTimeVector = list(desired_frame_interval * np.array(range(seq.shape[0])))
    ds.BitsAllocated = 8
    ds.PixelData = seq.tobytes()
    # pydicom.dcmwrite(dst_dcm_path, ds, write_like_original=False)
    seq = seq.transpose((2, 1, 0))
    nii_image = nib.Nifti1Image(seq, np.eye(4))
    nib.save(nii_image, dst_nii_path)

    seq = seq.transpose((2, 1, 0))
    '''2. Preparing MinIP images'''
    img_minip = np.min(seq, axis=0)
    img_minip = normalize(img_minip)
    dst_minip_path = os.path.join(input_paths.prepared_data_out_dir, mode, 'imgs_minip',
                                  '{}.png'.format(Path(dst_nii_path).stem))
    Path(dst_minip_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving minip to {}".format(dst_minip_path))
    imageio.imwrite(dst_minip_path, img_minip)
    # plt.imsave(dst_minip_path, img_minip, cmap=cm.gray)
    return len(cum_time_vector)


def prepare_masks(row, mode):  # mode = 'train', 'val', or 'test'
    """Prepare artery-vein segmentation ground-truth segmentations"""
    patient_id = row.patient_id

    '''1. Preparing artery mask'''
    artery_annotations = natsorted(glob.glob(
        os.path.join(input_paths.raw_artery_annotation_dir, patient_id,
                     '{}-{}-{}-labels.png'.format(row.filename, '[0-9]'*8, '[0-9]'*6))))
    if len(artery_annotations) != 0:
        if len(artery_annotations) > 1:
            logger.warning("Multiple arterial masks found for {} (taking the latest one): {}".format(patient_id, row.filename))
        artery_mask_path = artery_annotations[-1]
        artery_mask = cv.imread(artery_mask_path, cv.IMREAD_GRAYSCALE)
        dst_artery_mask_path = os.path.join(input_paths.prepared_data_out_dir, mode, 'masks_artery',
                                            '{}_{}.png'.format(patient_id, row.filename))
        logger.info("Copying mask: from {} to {}".format(artery_mask_path, dst_artery_mask_path))
        # cmap = colors.ListedColormap(['white', 'red', 'blue', 'purple'])
        # plt.imsave(dst_av_mask_path, av_mask, cmap=cmap)
        Path(dst_artery_mask_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(dst_artery_mask_path, artery_mask)
    else:
        logger.error("Arterial mask not found for {}: {}".format(patient_id, row.filename))

    '''2. Preparing vein mask'''
    vein_annotations = natsorted(glob.glob(
        os.path.join(input_paths.raw_vein_annotation_dir, patient_id,
                     '{}-{}-{}-labels.png'.format(row.filename, '[0-9]'*8, '[0-9]'*6))))
    if len(vein_annotations) != 0:
        if len(vein_annotations) > 1:
            logger.warning("Multiple venous masks found for {} (taking the latest one): {}".format(patient_id, row.filename))
        vein_mask_path = vein_annotations[-1]
        vein_mask = cv.imread(vein_mask_path, cv.IMREAD_GRAYSCALE)
        dst_vein_mask_path = os.path.join(input_paths.prepared_data_out_dir, mode, 'masks_vein',
                                          '{}_{}.png'.format(patient_id, row.filename))
        logger.info("Copying venous mask: from {} to {}".format(vein_mask_path, dst_vein_mask_path))
        Path(dst_vein_mask_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(dst_vein_mask_path, vein_mask)
    else:
        logger.error("Venous mask not found for {}: {}".format(patient_id, row.filename))

    '''3. Preparing vessel mask'''
    if len(artery_annotations) * len(vein_annotations) != 0:
        vessel_mask = np.zeros_like(artery_mask, dtype=np.uint8)
        vessel_mask[artery_mask > 0] = 1
        vessel_mask[vein_mask > 0] = 1
        dst_vessel_mask_path = os.path.join(input_paths.prepared_data_out_dir, mode, 'masks_vessel',
                                            '{}_{}.png'.format(patient_id, row.filename))
        logger.info("Saving binary vessel mask to {}".format(dst_vessel_mask_path))
        Path(dst_vessel_mask_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(dst_vessel_mask_path, vessel_mask)

    '''3. Preparing artery-vein mask'''
    if len(artery_annotations) * len(vein_annotations) != 0:
        av_mask = np.zeros_like(artery_mask, dtype=np.uint8)
        av_mask[artery_mask > 0] = 1
        av_mask[vein_mask > 0] = 2
        av_mask[np.multiply(artery_mask, vein_mask) > 0] = 3

        dst_av_mask_path = os.path.join(input_paths.prepared_data_out_dir, mode, 'masks_av',
                                        '{}_{}.png'.format(patient_id, row.filename))
        logger.info("Saving artery-vein vessel mask to {}".format(dst_av_mask_path))
        Path(dst_av_mask_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(dst_av_mask_path, av_mask)


if __name__ == '__main__':
    log_filepath = 'log/{}.log'.format(Path(__file__).stem)
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.FileHandler(log_filepath, mode='w'), logging.StreamHandler(sys.stdout)])

    df_info = pd.read_csv(input_paths.patient_info_csv)
    patient_ids = df_info['patient_id'].unique().tolist()
    random.seed(0)
    test_patients = random.sample(patient_ids, round(0.3 * len(patient_ids)))
    train_val_patients = [p for p in patient_ids if p not in test_patients]
    train_patients = random.sample(train_val_patients, round(0.5 * len(patient_ids)))

    df_test = df_info[df_info['patient_id'].isin(test_patients)]
    df_train = df_info[df_info['patient_id'].isin(train_patients)]
    df_val = df_info[~df_info['patient_id'].isin(train_patients + test_patients)]

    '''Prepare sequences'''
    raw_seq_lens = []
    for idx, row in enumerate(df_train.itertuples()):
        logger.info("Preparing training set -- {}/{}, patient: {}".format(idx+1, len(df_train), row.patient_id))
        raw_seq_lens.append(prepare_sequence_and_minip(row, mode='train'))
        prepare_masks(row, mode='train')
    for idx, row in enumerate(df_val.itertuples()):
        logger.info("Preparing validation set -- {}/{}, patient: {}".format(idx+1, len(df_val), row.patient_id))
        raw_seq_lens.append(prepare_sequence_and_minip(row, mode='val'))
        prepare_masks(row, mode='val')

    for idx, row in enumerate(df_test.itertuples()):
        logger.info("Preparing test set -- {}/{}, patient: {}".format(idx+1, len(df_test), row.patient_id))
        raw_seq_lens.append(prepare_sequence_and_minip(row, mode='test'))
        prepare_masks(row, mode='test')

    logger.info("sequence length -- min: {}, max: {}, mean: {}".format(min(raw_seq_lens), max(raw_seq_lens), sum(raw_seq_lens) / len(raw_seq_lens)))
    logger.info("done!")
