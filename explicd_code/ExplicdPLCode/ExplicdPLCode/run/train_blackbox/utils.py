from kltn_utils import dataset as kltn_utils_dataset
from torch.utils.data import DataLoader


def load_dataset(config, transform, mode):
    dataset = kltn_utils_dataset.ImageDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
        transform=transform,
        class_names=config.class_names,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader
