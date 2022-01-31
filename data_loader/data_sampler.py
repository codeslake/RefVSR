"""
Modified from torch.utils.data.distributed.DistributedSampler
Support enlarging the dataset for *iteration-oriented* training, for saving time when restart the
dataloader after each epoch
"""
import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import numpy as np


class DistIterSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, ratio=1, is_train = True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.is_train = is_train
        self.dataset = dataset
        self.num_replicas = num_replicas # numer of multi GPUs
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        if is_train is False:
            self.idx_frame_acc = self.dataset.idx_frame_acc
            max_length = 0

            for rank in range(0, self.num_replicas):
                if rank > len(self.idx_frame_acc):
                    return iter([])
                indices = self.idx_frame_acc[rank:len(self.idx_frame_acc):self.num_replicas][:]
                indices = sum(indices, [])

                if max_length < len(indices):
                    max_length = len(indices)
            self.max_length = max_length

        # print('!!!!!!', len(self.dataset), self.total_size, self.num_replicas, self.num_samples)

    def __iter__(self):

        if self.is_train:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(int(self.epoch))
            indices = torch.randperm(self.total_size, generator=g).tolist()

            dsize = len(self.dataset)
            indices = [v % dsize for v in indices]

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            return iter(indices)

        else:
            rank = self.rank
            if rank > len(self.idx_frame_acc):
                return iter([])
            indices = self.idx_frame_acc[rank:len(self.idx_frame_acc):self.num_replicas][:]
            indices = np.array(sum(indices, []))
            #print('\n\nrank: ', rank, ' indices: ', indices, '\n\n\n')

            return iter(indices)

    def __len__(self):
        if self.is_train:
            return self.num_samples
        else:
            return self.max_length

    def set_epoch(self, epoch):
        self.epoch = epoch
