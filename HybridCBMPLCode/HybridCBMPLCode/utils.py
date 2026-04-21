import torch
from kltn_utils import kltn_utils
from torch.utils.data import DataLoader, TensorDataset

from .datasets.dataset import CustomDataset
from .models.conceptBank.hybrid_bank import HybridConceptBank


def get_dataset_for_img_feat(dataloader, clip_model, clip_model_name):
    res_img_feat = []
    res_label = []

    for img, label in dataloader:
        img_feat = kltn_utils.get_img_feat_from_clip_model(
            clip_model, clip_model_name, img
        )
        res_img_feat.append(img_feat)
        res_label.append(label)

    res_img_feat = torch.cat(res_img_feat, dim=0)
    res_label = torch.cat(res_label, dim=0)

    res_dataset = TensorDataset(res_img_feat, res_label)

    return res_dataset, res_img_feat


def load_val(config, clip_model, mode):
    train_transforms, val_transforms = kltn_utils.build_transform(config)

    valset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/{mode}",
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
    valset, _ = get_dataset_for_img_feat(valLoader, clip_model, config.clip_model)
    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    return valLoader


def load_train(config, clip_model):
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
        pin_memory=True,
    )
    trainset, train_img_feat = get_dataset_for_img_feat(
        trainLoader, clip_model, config.clip_model
    )
    trainLoader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    return trainLoader, train_img_feat


def print_first_batch(dataloader):
    return_tuple = next(dataloader)

    for item in return_tuple:
        kltn_utils.rank_zero_info_newline(f"{item.shape}")


def load_concept_bank(config, train_img_feat):
    concept_bank = HybridConceptBank(config)

    concept_bank.initialize(
        img_features=train_img_feat,
        num_images_per_class=config.num_images_per_class,
        captions=None,
    )

    return concept_bank
