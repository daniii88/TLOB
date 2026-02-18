from typing import Optional

import torch
from torch.utils import data
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
import numpy as np
import constants as cst

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y, seq_size):
        """Initialization""" 
        self.seq_size = seq_size
        self.x = x
        self.y = y
        if type(self.x) == np.ndarray:
            self.x = torch.from_numpy(x).float()
        if type(self.y) == np.ndarray:
            self.y = torch.from_numpy(y).long()
        max_windows = max(int(self.x.shape[0]) - int(self.seq_size) + 1, 0)
        self.length = min(int(self.y.shape[0]), max_windows)
        self.data = self.x

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i+self.seq_size, :]
        return input, self.y[i]


def safe_collate(batch):
    """Collate without shared-memory resize path that breaks on non-resizable storages."""
    inputs, labels = zip(*batch)
    inputs = torch.stack(tuple(x.contiguous() for x in inputs), dim=0)
    labels = torch.stack(tuple(y for y in labels), dim=0)
    return inputs, labels


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_set,
        val_set,
        batch_size,
        test_batch_size,
        is_shuffle_train=True,
        test_set=None,
        num_workers=16,
        train_sampler: Optional[Sampler] = None,
    ):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.is_shuffle_train = is_shuffle_train
        self.train_sampler = train_sampler
        if train_set.data.device.type != cst.DEVICE:       #this is true only when we are using a GPU but the data is still on the CPU
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.num_workers = num_workers

    def train_dataloader(self):
        use_shuffle = self.is_shuffle_train and self.train_sampler is None
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=use_shuffle,
            sampler=self.train_sampler,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=safe_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=safe_collate,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=safe_collate,
        )

        
    
