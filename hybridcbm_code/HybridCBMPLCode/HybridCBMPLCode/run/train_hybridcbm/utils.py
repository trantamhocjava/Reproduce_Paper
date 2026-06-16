import os

from kltn_utils import dataset, kltn_utils
from kltn_utils.cbm import const as cbm_const
from torch.utils.data import DataLoader


def load_dataloader(
    config,
    transform,
    concept2class,
    mode,
):
    data_set = dataset.ImageConceptDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
        transform=transform,
        class_names=config.class_names,
        concept2class=concept2class,
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=config.batch_size,
        shuffle=True if mode == "train" else False,
        num_workers=4,
        pin_memory=True,
    )

    return data_loader


def setup_train(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(config.cp_path, exist_ok=True)

    config.class_concept = cbm_const.CLASS_AND_CONCEPT[config.dataset_name]
    config.class_names = config.class_concept["class_names"]
    config.num_class = len(config.class_names)


def load_dataset(
    dataset_dir,
    class_names,
    transform,
    concept2class,
    mode,
):
    data_set = dataset.ImageConceptDataset(
        dataset_dir=f"{dataset_dir}/{mode}",
        transform=transform,
        class_names=class_names,
        concept2class=concept2class,
    )

    return data_set
