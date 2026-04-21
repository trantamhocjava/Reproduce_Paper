from kltn_utils import kltn_utils
from torch.utils.data import DataLoader

from .dataset.dataset import CustomDataset, CustomDatasetForBlackbox


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label, concept = next(iter(loader))

    kltn_utils.rank_zero_info_newline(f"data shape: {data.shape}")
    kltn_utils.rank_zero_info_newline(f"label shape: {label.shape}")
    kltn_utils.rank_zero_info_newline(f"concept shape: {concept.shape}")


def load_train_val_test(config, class2concept):
    train_transforms, val_transforms = kltn_utils.build_transform(config)

    trainset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/train",
        config=config,
        class2concept=class2concept,
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
        class2concept=class2concept,
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
        class2concept=class2concept,
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


def load_train_val_test_for_blackbox(config):
    train_transforms, val_transforms = kltn_utils.build_transform(config)

    trainset = CustomDatasetForBlackbox(
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

    valset = CustomDatasetForBlackbox(
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

    testset = CustomDatasetForBlackbox(
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
