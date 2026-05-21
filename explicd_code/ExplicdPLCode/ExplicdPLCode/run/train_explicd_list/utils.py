from kltn_utils import dataset
from torch.utils.data import DataLoader


def load_dataset(config, transform, mode):
    data_set = dataset.ImageConceptDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
        transform=transform,
        class_names=config.class_names,
        concept2class=config.concept2class,
    )

    dataloader = DataLoader(
        dataset=data_set,
        batch_size=config.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader
