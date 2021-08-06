import deepspeed.runtime.zero.stage2 as stage2
import deepspeed.runtime.zero.stage3 as stage3
from turing.models import BertMultiTask

import os
import sys
import time
import logging
import numpy as np
import random
import json
import torch
from torch.cuda import nvtx
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from turing.logger import Logger
from turing.utils import get_sample_writer
from turing.models import BertMultiTask
from turing.dataset import PreTrainingDataset, PretrainBatch, PretrainDataType
from turing.sources import PretrainingDataCreator, WikiPretrainingDataCreator, TokenInstance
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear, warmup_linear_decay_exp, warmup_exp_decay_exp, warmup_exp_decay_poly
from utils import get_argument_parser, is_time_to_exit


import deepspeed

def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))

    # choose dataset and training config based on the given sequence length
    seq_len = str(args.max_seq_length)

    datasets = config["data"]["mixed_seq_datasets"][seq_len]
    del config["data"]["mixed_seq_datasets"]
    training = config["mixed_seq_training"][seq_len]
    del config["mixed_seq_training"]
    config["data"]["datasets"] = datasets
    config["training"] = training
    args.config = config

    args.job_name = config['name'] if args.job_name is None else args.job_name
    print("Running Config File: ", args.job_name)
    # Setting the distributed variables
    print("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                         args.job_name)

    args.n_gpu = 1

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    # Set validation dataset path
    if args.validation_data_path_prefix is None:
        logging.warning(
            'Skipping validation because validation_data_path_prefix is unspecified'
        )

    # Issue warning if early exit from epoch is configured
    if args.max_steps < sys.maxsize:
        logging.warning(
            'Early training exit is set after {} global steps'.format(
                args.max_steps))

    if args.max_steps_per_epoch < sys.maxsize:
        logging.warning('Early epoch exit is set after {} global steps'.format(
            args.max_steps_per_epoch))

    return args

def main():
    """"""
    args = construct_arguments()
    model = BertMultiTask(args)
    print('zero2 memory estimate')
    stage2.estimate_zero2_model_states_mem_needs_all_live(model.network, num_gpus_per_node=8, num_nodes=4)
    print('zero3 memory est. num node 1')
    stage3.estimate_zero3_model_states_mem_needs_all_live(model.network, num_gpus_per_node=8, num_nodes=1)
    print('zero3 num node 4')
    stage3.estimate_zero3_model_states_mem_needs_all_live(model.network, num_gpus_per_node=8, num_nodes=4)

if __name__ == "__main__":
    main()