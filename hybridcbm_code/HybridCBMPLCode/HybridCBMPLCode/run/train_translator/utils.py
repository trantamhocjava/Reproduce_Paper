from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import MergeDataset


def load_dataset(config, mode):
    dataset = MergeDataset(
        conceptPath=f"{config.concept_dir}/{mode}",
        cocoPath=f"{config.coco_dir}/{mode}",
        max_seq_len=config.max_seq_len,
    )

    sampler = None
    shuffle = False
    if mode == "train":
        sampler = DistributedSampler(dataset)
        shuffle = True

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

    return dataloader
