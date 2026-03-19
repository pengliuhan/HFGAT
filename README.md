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

We utilize a Tesla V100-SXM2-16GB GPU for testing.
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
