import lmdb
import os.path as op
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import random

def make_lmdb_from_imgs(img_dir,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        map_size=None):

    # check
    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')

    assert lmdb_path.endswith('.lmdb'), "lmdb_path must end with '.lmdb'."
    assert not op.exists(lmdb_path), f'Folder {lmdb_path} already exists. Exit.'

    # display info
    num_img = len(img_path_list)

    all_image_list = concat_image(img_dir, img_path_list)
    if map_size is None:
        img = cv2.imread(all_image_list[0], cv2.IMREAD_UNCHANGED)
        h_lq, w_lq, _ = img.shape
        _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
        )

        data_size_per_img = img_byte.nbytes
        data_size = data_size_per_img * len(img_path_list)*5
        map_size = data_size * 10  # enlarge the estimation

    # create lmdb environment & write data to lmdb
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    txt_file = open(op.join(lmdb_path, 'meta_info.txt'), 'w')
    pbar = tqdm(total=num_img, ncols=80)

    # The original images were divided into five 240 × 240 patches and saved separately to speed up training data loading.
    for cdx in range(5):
        for idx, (path, key) in enumerate(zip(img_path_list, keys)):
            aa,bb,cc = key.split('/')
            aa = int(aa)+108*cdx
            aa = "%03d" % (aa)
            key = f'{aa}/{bb}/{cc}'

            pbar.set_description(f'Write {key}')
            pbar.update(1)

            _, img_byte, img_shape = _read_img_worker(
                all_image_list[idx], key, compress_level,cdx
            )  # use _read function
            h, w, c = img_shape
            # write lmdb
            key_byte = key.encode('ascii')
            txn.put(key_byte, img_byte)

            # write meta
            txt_file.write(f'{key} ({h},{w},{c}) {compress_level}\n')

            # commit per batch
            if idx % batch == 0:
                txn.commit()
                txn = env.begin(write=True)

    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()


def concat_image(img_dir, img_path_list):
    length = len(img_dir)
    all_list = []
    for idx in range(length):
        all_list.append(img_dir[idx] + '/' + "%04d" % (img_path_list[idx]) + '.png')

    return all_list


def _read_img_worker(path, key, compress_level, index):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    H, W, C = img.shape
    if index==0:
        img = img[:240,:240,:]
    elif index == 1:
        img = img[H-240:, :240, :]
    elif index==2:
        img = img[H-240:, W-240:, :]
    elif index==3:
        img = img[:240,W-240:,:]
    elif index==4:
        img = img[int(H/2)-120:int(H/2)+120, int(W/2)-120:int(W/2)+120, :]

    h, w, c = img.shape
    _, img_byte = cv2.imencode(
        '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    )

    return (key, img_byte, (h, w, c))
