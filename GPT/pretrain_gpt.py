import json
import sys
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (DataCollatorForLanguageModeling, 
                        GPT2Tokenizer, GPT2LMHeadModel, GPT2Config)

def count_parameters(model):
    return sum(p.ds_numel for p in model.parameters())


def print_at_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)

def get_args():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--num-epochs", help="epochs", default=10, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--job_name", type=str)
    deepspeed.add_config_arguments(parser)
    args = parser.parse_args(sys.argv[1:])

    config = json.load(open(args.deepspeed_config, 'r', encoding='utf-8'))
    args.config = config

    return args


class FakeDataset(Dataset):
    """temporarily using this to avoid having to load a real dataset"""

    def __init__(self, epoch_sz: int) -> None:
        self.__epoch_sz = epoch_sz

    def __getitem__(self, _: int) -> dict:
        return {"text": "Hello, my dog is cute, this is a fake sentence"}

    def __len__(self) -> int:
        return self.__epoch_sz


class CollatorForLMWrapper(DataCollatorForLanguageModeling):
    def __init__(self, device, max_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.max_length = max_length

    def __call__(self, examples):
        batch = self.tokenizer(
            [e["text"] for e in examples],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt"
        )

        batch = list(
            map(dict, zip(*[[(k, v) for v in batch[k]] for k in batch.keys()]))
        )

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.cuda()
        return batch

def get_dataloader(args, tokenizer): 
    train_dataset = FakeDataset(200000)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    micro_bs = args.config['train_micro_batch_size_per_gpu']

    training_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        collate_fn=CollatorForLMWrapper(
                tokenizer=tokenizer,
                device=torch.device(f"cuda:{args.local_rank}"),
                max_length=512,
                mlm=False,
        ),
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=1234,
        ),
        )

    return training_loader

def get_model(args) -> Module:
    model_config = args.config['gpt_config']
    cfg = GPT2Config(
        vocab_size=model_config['vocab_size'],
        n_positions=model_config['max_position_embeddings'],
        n_embd=model_config['embedding_dim'],
        n_layer=model_config['num_hidden_layers'],
        n_head=model_config['num_attention_heads'],
        n_inner=model_config['intermediate_size'],
        use_cache=False if model_config['gradient_checkpointing'] else True
    )

    if args.config['zero_optimization']['stage'] == 3:
        with deepspeed.zero.Init(config=args.config):
            model = GPT2LMHeadModel(cfg)
    else:
        model = GPT2LMHeadModel(cfg)

    print_at_rank0(model)
    print_at_rank0(f"model param size {count_parameters(model)/1e9} B")
    if model_config['gradient_checkpointing'] == True:
        model.gradient_checkpointing_enable()

    return model


def main():
    args = get_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f'tokenizer vocab_size {tokenizer.vocab_size}'
            f' config vocab size {args.config["gpt_config"]["vocab_size"]}')
    assert tokenizer.vocab_size <= args.config["gpt_config"]["vocab_size"]
    # get data_loader
    training_dataloader = get_dataloader(args, tokenizer)
    
    model = get_model(args)

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.config,
    )
    for e in range(args.num_epochs):
        for n, inputs in enumerate(training_dataloader):
            if n < 5 and e < 1:
                print_at_rank0(f"{[inputs[k].size() for k in inputs]}")
            outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs["input_ids"])
            loss = outputs.loss
            model.backward(loss) 
            model.step()
            
            if model.is_gradient_accumulation_boundary():
                print_at_rank0(f"{e} {n}, LOSS: {loss.item()}")

            loss = None


if __name__ == "__main__":
    main()

    