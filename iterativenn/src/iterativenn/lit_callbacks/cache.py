import os
import time
from typing import Any, Dict, List

import omegaconf
import pytorch_lightning as pl
import torch
from asteval import Interpreter
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
            self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        torch.save(prediction, os.path.join(self.output_dir, f"{dataloader_idx}", f"{batch_idx}.pt"))

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


class CacheReader(pl.Callback):
    def __init__(self, root_train, root_val, root_test, train_cache_size=16,
                 validation_cache_size=16, test_cache_size=16,
                 mode='full_in_memory', device='cpu'):
        super().__init__()
        self.train_cache = {'ids': [], 'values': []}
        self.validation_cache = {'ids': [], 'values': []}
        self.test_cache = {'ids': [], 'values': []}
        """
        Here cache ids are the batch_idx
        And cache values are the tensor values
        """

        self.root_train = root_train
        self.root_val = root_val
        self.root_test = root_test

        self.files_train = sorted(os.listdir(self.root_train), key=lambda x: int(x.split('.')[0]))
        self.files_val = sorted(os.listdir(self.root_val), key=lambda x: int(x.split('.')[0]))
        self.files_test = sorted(os.listdir(self.root_test),  key=lambda x: int(x.split('.')[0]))

        self.device = device

        if mode == 'full_in_memory':
            self.train_cache_size = len(self.files_train)
            self.validation_cache_size = len(self.files_val)
            self.test_cache_size = len(self.files_test)
        elif mode == 'fixed_cache_size':
            self.train_cache_size = train_cache_size  # multiple of batch_size is recommended
            # if train_cache_size is  len(self.files_train) then we load all data in cache
            self.validation_cache_size = validation_cache_size
            self.test_cache_size = test_cache_size
        else:
            raise ValueError(f"mode: {mode} is not supported")

    def load_data_in_cache(self, cache, root, files, cache_size, start=0):
        for i in range(start, start+cache_size):
            batch_idx = int(files[i].split('.')[0])  # remove .pt
            print(f"batch_idx: {batch_idx}")
            batch = torch.load(os.path.join(root, files[batch_idx])).requires_grad_(True)
            cache['ids'].append(batch_idx)  # batch_idx is the key
            cache['values'].append(batch.to(self.device))
        print(cache['ids'])
        return cache

    # def _clear_cache(self, cache):
    #     print("Clearing cache")
    #     cache.clear()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_cache = self.load_data_in_cache(self.train_cache, self.root_train, self.files_train,
                                                   self.train_cache_size)
        pl_module.train_cache = self.train_cache


        self.validation_cache = self.load_data_in_cache(self.validation_cache, self.root_val, self.files_val,
                                                        self.validation_cache_size)
        pl_module.validation_cache = self.validation_cache

        self.test_cache = self.load_data_in_cache(self.test_cache, self.root_test, self.files_test,
                                                  self.test_cache_size)
        pl_module.test_cache = self.test_cache


    def on_train_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if (batch_idx % self.train_cache_size) == 0 and batch_idx > 0:
            print(f"batch_idx: {batch_idx} clear cache")
            for k in self.train_cache.keys(): # reinitialize the cache
                self.train_cache[k].clear() # clear the cache

            # load new batch from files_train for next train_cache_size batches

            self.train_cache = self.load_data_in_cache(self.train_cache, self.root_train, self.files_train,
                                                       self.train_cache_size, start=batch_idx)

            pl_module.train_cache = self.train_cache
