from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import MergeDataset


def load_train(config):
    trainset = MergeDataset(config.conceptPath, config.cocoPath)

    sampler = DistributedSampler(trainset)

    trainLoader = DataLoader(
        trainset,
        sampler=sampler,
        batch_size=config.batch_size,
        num_workers=4,
        drop_last=True,
    )

    return trainLoader


def get_prefix_size(clip_model_name):
    if clip_model_name == "ViT-L/14":
        prefix_size = 768
    elif clip_model_name == "ViT-B/32":
        prefix_size = 512
    else:
        prefix_size = 1024

    return prefix_size
