import json

import clip
import numpy as np
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch import optim
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

from . import const
from .dataset.dataset import CustomDataset


def save_dict_to_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def seed_everything_in_pl():
    seed_everything(const.SEEDING, workers=True)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_optimizer(model, config):
    grad_true_param = filter(lambda p: p.requires_grad, model.parameters())

    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            grad_true_param,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd_v1":
        optimizer = optim.SGD(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            grad_true_param,
            lr=config.lr,
            betas=(config.beta_1, config.beta_2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam_v1":
        optimizer = optim.Adam(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    return optimizer


def build_transform(config):
    """Uniform preprocess follow preprocess architecture got from the code below
    ```
    from open_clip import create_model_from_pretrained, get_tokenizer

    _, preprocess = create_model_from_pretrained(
        "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    )
    ```

    """
    if config.transform == "uniform":
        train_transform = v2.Compose(const.PREPROCESS_LIST)
        val_transform = v2.Compose(const.PREPROCESS_LIST)

    return train_transform, val_transform


def build_scheduler(optimizer, config):
    if not config.use_scheduler:
        return None, None

    monitor = None
    if config.scheduler == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=0.01,
            total_iters=config.epochs,
        )
    elif config.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.decrease_every,
            gamma=1 / config.lr_divisor,
        )
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        monitor = "val_loss"

    return scheduler, monitor


def read_img(img_path):
    res = None

    try:
        res = read_image(
            img_path,
            mode=ImageReadMode.RGB,
        )
    except Exception:
        img = Image.open(img_path).convert("RGB")
        res = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)

    return res


def build_clip_model(clip_model_name):
    if clip_model_name in const.CLIP_MODEL_FROM_OPENAI:
        model, _ = clip.load(clip_model_name)
        tokenizer = clip.tokenize

    if (
        clip_model_name
        == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        model, _ = create_model_from_pretrained(clip_model_name)
        tokenizer = get_tokenizer(clip_model_name)

    return model, tokenizer


def load_train_val_test(config):
    train_transforms, val_transforms = build_transform(config)

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


def get_mode(monitor):
    if monitor in const.METRIC_MAX:
        return "max"
    else:
        return "min"
