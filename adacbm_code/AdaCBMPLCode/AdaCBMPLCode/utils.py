from kltn_utils import kltn_utils
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader

from .dataset.dataset import CustomDataset


def load_train_val_test(config):
    train_transforms, val_transforms = kltn_utils.build_transform(config)

    trainset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/train",
        config=config,
        transforms=train_transforms,
    )

    trainLoader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/val",
        config=config,
        transforms=val_transforms,
    )

    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    testset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/test",
        config=config,
        transforms=val_transforms,
    )
    testLoader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    return trainLoader, valLoader, testLoader


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label = next(iter(loader))

    rank_zero_info(f"data shape: {data.shape}")
    rank_zero_info(f"label shape: {label.shape}")
