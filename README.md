# *Hierarchical Frequency-Guided Alignment Transformer for Compressed Video Quality Enhancement*


The *PyTorch* implementation for the [Hierarchical Frequency-Guided Alignment Transformer for Compressed Video Quality Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/37784) which is accepted by [AAAI26].

Task: Video Quality Enhancement / Video Artifact Reduction.


## 1. Pre-request

### 1.1. Environment
```bash
conda create -n hfgat python=3.8 -y  
conda activate hfgat
git clone --depth=1 https://github.com/pengliuhan/HFGAT && cd HFGAT/
pip install -r requirements.txt
```

### 1.2. Dataset

Please check [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset).

### 1.3. Install FFmpeg

### 1.4. Create LMDB
We now generate LMDB to speed up IO during training.
```bash
python datasets/create_lmdb.py
```

## 2. Train

We utilize a Tesla V100-SXM2-16GB GPU for training.
```bash
python train.py
```

## 3. Test
1. Convert YUV format video to RGB format
```bash
def yuv_to_rgb(gt_dir, lq_dir):
    rgb_gt_dir = gt_dir.rstrip('/') + "_test_RGB/"
    rgb_lq_dir = lq_dir.rstrip('/') + "_test_RGB/"

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
```

2. We utilize a Tesla V100-SXM2-16GB GPU for testing.
```bash
python test.py
```

## Citation
If you find this project is useful for your research, please cite:
```bash
@article{Peng_Li_Gao_Ye_Lv_2026, 
  title={Hierarchical Frequency-Guided Alignment Transformer for Compressed Video Quality Enhancement}, 
  volume={40}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/37784}, 
  DOI={10.1609/aaai.v40i10.37784}, 
  number={10}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Peng, Liuhan and Li, Shuai and Gao, Yanbo and Ye, Mao and Lv, Chong}, 
  year={2026}, 
  month={Mar.}, 
  pages={8349-8357} 
}
```

## Acknowledgements
This work is based on [STDF-Pytoch](https://github.com/RyanXingQL/STDF-PyTorch), [Uformer](https://github.com/ZhendongWang6/Uformer) and [LINF](https://github.com/JNNNNYao/LINF).


