import json
import sys
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (DataCollatorForLanguageModeling, RobertaConfig,
                          RobertaForMaskedLM, RobertaTokenizerFast)


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
        return {"text": "this is a fake sentence"}

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
                mlm=True,
                mlm_probability=0.15,
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
    model_config = args.config['roberta_config']

    cfg = RobertaConfig(
        max_position_embeddings=model_config['max_position_embeddings'],
        type_vocab_size=model_config['type_vocab_size'],
        num_attention_heads=model_config['num_attention_heads'],
        num_hidden_layers=model_config['num_hidden_layers'],
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        gradient_checkpointing=model_config['gradient_checkpointing'],
        vocab_size=model_config['vocab_size'],
    )
    if args.config['zero_optimization']['stage'] == 3:
        with deepspeed.zero.Init(config=args.config):
            model = RobertaForMaskedLM(cfg)
    else:
        model = RobertaForMaskedLM(cfg)

    print_at_rank0(model)

    return model


def main():
    args = get_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    print(f'tokenizer vocab_size {tokenizer.vocab_size}'
            f' config vocab size {args.config["roberta_config"]["vocab_size"]}')
    assert tokenizer.vocab_size <= args.config["roberta_config"]["vocab_size"]
    # get data_loader
    training_dataloader = get_dataloader(args, tokenizer)
    
    model = get_model(args)

    # with deepspeed.zero.Init(config=ds_config):
    with torch.cuda.nvtx.range("create_training_engine"):
        with torch.cuda.nvtx.range("deepspeed.initialize"):
            model, _, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=args.config,
            )
    for e in range(args.num_epochs):
        for n, batch in enumerate(training_dataloader):
            if n < 5 and e < 1:
                # verify the input size
                print_at_rank0(f"{[batch[k].size() for k in batch]}")

            loss = model(batch["input_ids"],
                    batch["attention_mask"],
                    labels=batch["labels"],).loss
            model.backward(loss) 
            model.step()
            
            if model.is_gradient_accumulation_boundary():
                print_at_rank0(f"{e} {n}, LOSS: {loss.item()}")

            loss = None


if __name__ == "__main__":
    main()

    