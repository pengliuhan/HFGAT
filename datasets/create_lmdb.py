"""
Create LMDB for the training set of Vimeo-90K.

GT: 64,612 training sequences out of 91701 7-frame sequences.
LQ: HM16.5-intra-compressed sequences.
key: assigned from 00000 to 99999.

Sym-link Vimeo-90K dataset root to ./data/vimeo90k folder.
"""
import argparse
import os
import glob
import yaml
import os.path as op
from lmdb_datatset import make_lmdb_from_imgs

def concat_image(img_dir, img_path_list):
    length = len(img_dir)
    all_list = []
    for idx in range(length):
        all_list.append(img_dir[idx]+'/'+img_path_list[idx])
    return all_list

parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_path', type=str, default='../config/train.yaml',
    help='Path to option YAML file.'
    )
args = parser.parse_args()
yml_path = args.opt_path


def yuv_to_rgb(gt_dir, lq_dir):
    rgb_gt_dir = gt_dir.rstrip('/') + "_RGB/"
    rgb_lq_dir = lq_dir.rstrip('/') + "_RGB/"

    os.makedirs(rgb_gt_dir, exist_ok=True)
    os.makedirs(rgb_lq_dir, exist_ok=True)

    yuv_list = sorted([os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith('.yuv')])
    for name in yuv_list:
        os.makedirs(os.path.join(rgb_gt_dir, name), exist_ok=True)
        os.makedirs(os.path.join(rgb_lq_dir, name), exist_ok=True)

        _, wxh, nfs = name.split('_')
        w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])

        cmd_gt = 'ffmpeg -s '+str(w)+'x'+str(h)+' -i '+gt_dir+name+'.yuv '+rgb_gt_dir+name+'/%4d.png'
        cmd_lq = 'ffmpeg -s '+str(w)+'x'+str(h)+' -i '+lq_dir+name+'.yuv '+rgb_lq_dir+name+'/%4d.png'

        exit_code_gt = os.system(cmd_gt)
        exit_code_lq = os.system(cmd_lq)
        if exit_code_gt != 0:
            print(f"GT: {exit_code_gt}")
        if exit_code_lq != 0:
            print(f"LQ: {exit_code_lq}")

    return rgb_gt_dir, rgb_lq_dir



def create_lmdb_for_vimeo90k():
    # video info
    with open(yml_path, 'r') as fp:
        fp = yaml.load(fp, Loader=yaml.FullLoader)
        root_dir = fp['train_dataset']['root']
        gt_folder = fp['train_dataset']['gt_folder']
        lq_folder = fp['train_dataset']['lq_folder']
        gt_path = fp['train_dataset']['gt_path']
        lq_path = fp['train_dataset']['lq_path']
        radius = fp['train_dataset']['radius']
    gt_dir = op.join(root_dir, gt_folder)
    lq_dir = op.join(root_dir, lq_folder)
    rgb_gt_dir, rgb_lq_dir = yuv_to_rgb(gt_dir, lq_dir)  # 将yuv视频保存为RGB图片
    lmdb_gt_path = op.join(root_dir, gt_path)
    lmdb_lq_path = op.join(root_dir, lq_path)

    # scan all videos
    print('Scaning meta list...')
    gt_video_list = sorted([os.path.join(rgb_gt_dir, d) for d in os.listdir(rgb_gt_dir)
               if os.path.isdir(os.path.join(rgb_gt_dir, d))])

    lq_video_list = sorted([os.path.join(rgb_lq_dir, d) for d in os.listdir(rgb_lq_dir)
               if os.path.isdir(os.path.join(rgb_lq_dir, d))])

    msg = f'> {len(gt_video_list)} videos found.'
    print(msg)

    print("Scaning GT frames (only center frames of each sequence)...")
    frm_list = []
    for gt_video_path in gt_video_list:
        gtlist = os.listdir(gt_video_path)
        gtlist.sort()
        nfs = gtlist.__len__()
        num_seq = nfs // (2 * radius + 1)
        frm_list.append([radius+ 1 + iter_seq * (2 * radius + 1) for iter_seq in range(num_seq)])

    num_frm_total = sum([len(frms) for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)

    key_list = []
    video_path_list = []
    index_frame_list = []
    for iter_vid in range(len(gt_video_list)):
        frms = frm_list[iter_vid]
        for iter_frm in range(len(frms)):
            key_list.append('{:03d}/{:03d}/im4.png'.format(iter_vid+1, iter_frm+1))
            video_path_list.append(gt_video_list[iter_vid])
            index_frame_list.append(frms[iter_frm])

    print("Writing LMDB for GT data...")

    make_lmdb_from_imgs(
        img_dir=video_path_list,
        lmdb_path=lmdb_gt_path,
        img_path_list=index_frame_list,
        keys=key_list,
        multiprocessing_read=False
    )
    print("> Finish.")

    # # generate LMDB for LQ
    print("Scaning LQ frames...")
    len_input = 2 * radius + 1
    frm_list = []
    for lq_video_path in lq_video_list:
        lqlist = os.listdir(lq_video_path)
        lqlist.sort()
        nfs = lqlist.__len__()

        num_seq = nfs // len_input
        frm_list.append([list(range(iter_seq * len_input+1, (iter_seq + 1) \
            * len_input+1)) for iter_seq in range(num_seq)])

    num_frm_total = sum([len(frms) * len_input for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)
    key_list = []
    video_path_list = []
    index_frame_list = []

    for iter_vid in range(len(lq_video_list)):
        frm_seq = frm_list[iter_vid]
        for iter_seq in range(len(frm_seq)):
            key_list.extend(['{:03d}/{:03d}/im{:d}.png'.format(iter_vid+1, \
                iter_seq+1, i) for i in range(1, len_input+1)])
            video_path_list.extend([lq_video_list[iter_vid]] * len_input)
            index_frame_list.extend(frm_seq[iter_seq])

    print("Writing LMDB for LQ data...")
    make_lmdb_from_imgs(
        img_dir=video_path_list,
        lmdb_path=lmdb_lq_path,
        img_path_list=index_frame_list,
        keys=key_list,
        multiprocessing_read=False,
    )
    print("> Finish.")

    # sym-link
    if not op.exists('data/MFQEv2'):
        if not op.exists('data/'):
            os.system("mkdir data/")
        os.system(f"ln -s {root_dir} ./data/MFQEv2")
        print("Sym-linking done.")
    else:
        print("data/MFQEv2 already exists.")
    

if __name__ == '__main__':
    create_lmdb_for_vimeo90k()
