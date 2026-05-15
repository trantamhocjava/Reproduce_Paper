from torch.utils.data import DataLoader

from ... import const
from .dataset import ImageDataset


def update_config_for_train(config):
    config.epochs = config.end_epoch - config.start_epoch + 1
    config.class_names = const.CLASS_AND_CONCEPT[config.dataset_name]["class_names"]
    config.num_class = len(config.class_names)

    return config


def load_dataloader(
    config,
    transform,
    concept2class,
    mode,
):
    dataset = ImageDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
        transforms=transform,
        config=config,
        concept2class=concept2class,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader
