import copy
from torch.utils.data import DataLoader as DataLoader
from torch.utils.data.sampler import Sampler
import math
import torch
from functools import partial
import random
import numpy as np

datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset


class DistSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch.

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(
            self, dataset, num_replicas=None, rank=None, ratio=1
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # enlarged by ratio, and then divided by num_replicas
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on ite_epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        # enlarge indices
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # ==========subsample
        # e.g., self.rank=1, self.total_size=4, self.num_replicas=2
        # indices = indices[1:4:2] = indices[i for i in [1, 3]]
        # for the other worker, indices = indices[i for i in [0, 2]]
        # ==========
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For shuffling data at each epoch. See train.py."""
        self.epoch = epoch

class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


def create_dataloader(
        dataset, opts_dict, sampler=None, phase='train', seed=None
):
    """Create dataloader."""

    if phase == 'train':
        # >I don't know why BasicSR have to detect `is_dist`
        dataloader_args = dict(
            dataset=dataset,
            batch_size=opts_dict['train_dataset']['batch_size'],
            shuffle=False,  # sampler will shuffle at train.py
            num_workers=opts_dict['train_dataset']['num_worker_per_gpu'],
            sampler=sampler,
            drop_last=True,
            pin_memory=True
        )
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            _worker_init_fn,
            num_workers=opts_dict['train_dataset']['num_worker_per_gpu'],
            rank=0,
            seed=seed
        )
    elif phase == 'val':
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

    return DataLoader(**dataloader_args)


def _worker_init_fn(worker_id, num_workers, rank, seed):
    # func for torch.utils.data.DataLoader
    # set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)