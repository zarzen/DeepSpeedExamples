import os
import random
import h5py
import logging
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import torch
from torch._C import device
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler

from bert_dataset_provider import BertDatasetProviderInterface
from turing.dataset import BatchType, map_to_torch

# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(input_file, max_predictions_per_seq,
                               num_workers, train_batch_size, worker_init,
                               data_sampler, use_customized=False):
    if not use_customized:
        train_data = pretraining_dataset(
            input_file=input_file, max_predictions_per_seq=max_predictions_per_seq)
    else:
        train_data = CustomizedDataset(
            input_file=input_file)

    train_dataloader = DataLoader(train_data,
                                  sampler=data_sampler(train_data),
                                  batch_size=train_batch_size,
                                  num_workers=num_workers,
                                  worker_init_fn=worker_init,
                                  pin_memory=True)
    return train_dataloader, len(train_data)


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_predictions_per_seq):
        self.input_file = input_file
        self.max_predictions_per_seq = max_predictions_per_seq
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else
            torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_predictions_per_seq
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            map_to_torch([BatchType.PRETRAIN_BATCH]), input_ids, input_mask,
            segment_ids, next_sentence_labels, masked_lm_labels
        ]


class CustomizedDataset(Dataset):
    r"""customized dataset is stored in HDF5 format
    and has four inputs: ``input_ids``, ``valid_length``, ``masked_lm_positions``,
    and ``masked_lm_ids``.

    Arguments:
        input_file (str): path to the data file.
    """

    def __init__(self, input_file):
        super(CustomizedDataset, self).__init__()
        self.input_file = input_file
        f = h5py.File(self.input_file, "r")
        # This is used to ensure that the previous generated datasets can still be used.
        # FIXME: Will remove this after we regenerated the data.
        if "input_mask" in f:
            self.has_mask = True
            keys = ["input_ids", "input_mask", "masked_lm_positions", "masked_lm_ids"]
        else:
            self.has_mask = False
            keys = ["input_ids", "valid_length", "masked_lm_positions", "masked_lm_ids"]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, x, masked_lm_positions, masked_lm_ids] = [
            torch.from_numpy(feature[index].astype(np.int64))
            for _, feature in enumerate(self.inputs)
        ]

        masked_lm_labels = (
            torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            * -1
        )
        index = masked_lm_positions.shape[-1]
        # store number of masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        # if self.has_mask:
        #     valid_length = x.sum(-1)
        # else:
        #     valid_length = x
        valid_length = x
        # make the data format align
        segment_ids = torch.zeros_like(input_ids, dtype=torch.int64, 
                                        device=input_ids.device)
        next_sentence_labels = torch.zeros(1, dtype=torch.int64, 
                                            device=input_ids.device)
        return [map_to_torch([BatchType.PRETRAIN_BATCH]), 
                input_ids, valid_length, segment_ids, next_sentence_labels, masked_lm_labels]


class NvidiaBertDatasetProvider(BertDatasetProviderInterface):
    def __init__(self, args):
        self.num_workers = args.config['training']['num_workers']
        self.max_seq_length = args.max_seq_length
        self.max_predictions_per_seq = args.max_predictions_per_seq

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
        self.logger = args.logger

        if args.local_rank == -1:
            self.global_rank = 0
            self.world_size = 1
        else:
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # Initialize dataset files
        dataset_path = os.path.join(
            args.data_path_prefix,
            args.config['data']['datasets']['pretrain_dataset'])
        self.dataset_files = [
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
            os.path.isfile(os.path.join(dataset_path, f)) and 'training' in f
        ]
        self.dataset_files.sort()
        random.shuffle(self.dataset_files)
        self.num_files = len(self.dataset_files)
        self.data_sampler = RandomSampler

        self.worker_init = WorkerInitObj(args.seed + args.local_rank)
        self.dataset_future = None
        self.pool = ProcessPoolExecutor(1)

        if self.global_rank == 0:
            self.logger.info(
                f"NvidiaBertDatasetProvider - Initialization:  num_files = {self.num_files}"
            )
        self.use_customized = args.use_customized_data

    def get_shard(self, index):
        if self.dataset_future is None:
            data_file = self._get_shard_file(index)
            self.train_dataloader, sample_count = create_pretraining_dataset(
                input_file=data_file,
                max_predictions_per_seq=self.max_predictions_per_seq,
                num_workers=self.num_workers,
                train_batch_size=self.train_micro_batch_size_per_gpu,
                worker_init=self.worker_init,
                data_sampler=self.data_sampler,
                use_customized=self.use_customized)
        else:
            self.train_dataloader, sample_count = self.dataset_future.result(
                timeout=None)

        return self.train_dataloader, sample_count

    def release_shard(self, index):
        del self.train_dataloader

    def prefetch_shard(self, index):
        data_file = self._get_shard_file(index)
        self.dataset_future = self.pool.submit(
            create_pretraining_dataset, data_file,
            self.max_predictions_per_seq, self.num_workers,
            self.train_micro_batch_size_per_gpu, self.worker_init,
            self.data_sampler)

    def get_batch(self, batch_iter):
        return batch_iter

    def prefetch_batch(self):
        pass

    def _get_shard_file(self, shard_index):
        file_index = self._get_shard_file_index(shard_index, self.global_rank)
        return self.dataset_files[file_index % self.num_files]

    def _get_shard_file_index(self, shard_index, global_rank):
        if dist.is_initialized() and self.world_size > self.num_files:
            remainder = self.world_size % self.num_files
            file_index = (shard_index * self.world_size) + global_rank + (
                remainder * shard_index)
        else:
            file_index = shard_index * self.world_size + global_rank

        return file_index % self.num_files
