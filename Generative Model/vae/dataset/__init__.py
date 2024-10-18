import torch.utils.data as data

from .cifar import CifarDataset
from .mnist import MnistDataset
from .celebA import CelebADataset

def build_dataset(args, is_train=False):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.img_dim = 3
        return CifarDataset(is_train)
    
    elif args.dataset =='mnist':
        args.num_classes = 10
        args.img_dim = 1
        return MnistDataset(is_train)
    
    elif args.dataset == 'celebA':
        args.num_classes = 1
        args.img_dim = 3
        return CelebADataset(is_train)
    
    raise NotImplementedError(f"Unkonwn dataset: {args.dataset}")
                              
def build_dataloader(args, dataset, is_train=False):
    if is_train:
        sample = data.distributed.DistributedSampler(dataset) if args.distributed else data.RandomSampler(dataset)
        batch_sampler_train = data.BatchSampler(sample, args.batch_size // args.world_size, drop_last=True if is_train else False)
        dataloader = data.DataLoader(dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=False)
    else:
        dataloader = data.DataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=args.num_workers)

    return dataloader