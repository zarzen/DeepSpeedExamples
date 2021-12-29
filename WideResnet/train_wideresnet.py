import json
import sys
from argparse import ArgumentParser

import deepspeed
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.models.resnet import ResNet, Bottleneck

def count_parameters(model):
    try:
        return sum(p.ds_numel for p in model.parameters())
    except:
        return sum(p.numel() for p in model.parameters())


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

    def __init__(self, epoch_sz: int, W=224, H=224) -> None:
        self.__epoch_sz = epoch_sz
        self.W = W
        self.H = H

    def __getitem__(self, _: int) -> dict:
        return torch.rand((3, self.H, self.W)), torch.randint(0, 1000, (1,))

    def __len__(self) -> int:
        return self.__epoch_sz



def get_dataloader(args): 
    train_dataset = FakeDataset(8192 * 100)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    micro_bs = args.config['train_micro_batch_size_per_gpu']
    dtype = torch.half if args.config['fp16']['enabled'] else torch.float

    def _collate_fn(batch):
        # print_at_rank0(f"collate batch input {[e[0].size() for e in batch]}")
        data = torch.vstack([e[0].unsqueeze(0) for e in batch]).to(
            args.local_rank, dtype=dtype)
        labels = torch.vstack([e[1].unsqueeze(0) for e in batch]).to(
            args.local_rank)
        return data, labels

    training_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        collate_fn=_collate_fn,
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=1234,
        ),
        )

    return training_loader

def get_model(args) -> Module:
    model_config = args.config['wideresnet_config']
    block = Bottleneck
    layers = model_config['layers']
    kwargs = {}
    kwargs['width_per_group'] = model_config['width_per_group']
    model = ResNet(block, layers, **kwargs)

    print_at_rank0(model)
    print_at_rank0(f"model param size {count_parameters(model)/1e9} B")

    return model


def main():
    args = get_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')

    # get data_loader
    training_dataloader = get_dataloader(args)
    
    model = get_model(args)
    loss_fn = torch.nn.CrossEntropyLoss()

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=args.config,
    )
    for e in range(args.num_epochs):
        for n, inputs in enumerate(training_dataloader):
            if n < 5 and e < 1:
                print_at_rank0(f"inputs sizes {[e.size() for e in inputs]}, device {[e.device for e in inputs]}")
            outputs = model(inputs[0])
            loss = loss_fn(outputs, inputs[1].squeeze())
            model.backward(loss) 
            model.step()
            
            if model.is_gradient_accumulation_boundary():
                print_at_rank0(f"{e} {n}, LOSS: {loss.item()}")

            loss = None


if __name__ == "__main__":
    main()

    