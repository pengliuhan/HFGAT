import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
from datasets import register

def generate_it(t=0, nf=7, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f - 1).tolist()
    return index

@register('video-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.len_data = self.__len__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lq,img_gt = self.dataset[idx]
        img_lqs = []
        for cdx in generate_it(idx, 7, self.len_data):
            img_lq, _ = self.dataset[cdx]
            img_lqs.append(img_lq)

        img_lqs = torch.stack(img_lqs, dim=0)
        return {
            'lq': img_lqs,
            'gt': img_gt
        }
