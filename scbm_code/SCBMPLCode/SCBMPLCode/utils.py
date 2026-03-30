import ast
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import const
from .const import DATASET_CLASS
from .model.cbm import CBM
from .model.scbm import SCBM


def str2obj(text):
    return ast.literal_eval(text)


def load_img2attr(config):
    if config.dataset_name == "cub":
        img2attr = pd.read_csv(f"{config.dataset_dir}/img2selected_attr.csv")
        img2attr["attribute_label"] = img2attr["attribute_label"].map(
            lambda x: torch.tensor(str2obj(x))
        )

        img2attr["uncertain_attribute_label"] = img2attr[
            "uncertain_attribute_label"
        ].map(lambda x: torch.tensor(str2obj(x)))
        img2attr = img2attr.set_index("splitted_path")
    elif config.dataset_name == "awa2":
        img2attr = torch.tensor(
            np.load(f"{config.dataset_dir}/classes_attr_matrix.npy")
        )

    return img2attr


def create_model(config):
    """
    Parse the configuration file and return a relevant model
    """
    if config.model == "cbm":
        model = CBM(config)
    elif config.model == "scbm":
        model = SCBM(config)

    if config.compile:
        model = torch.compile(model)

    return model


def build_transform(config, preprocess_list):
    if config.transform == "paper":
        train_transform = v2.Compose(
            [
                v2.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                v2.RandomResizedCrop(299),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        val_transform = v2.Compose(
            [
                v2.CenterCrop(299),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif config.transform == "follow_backbone":
        train_transform = v2.Compose(preprocess_list)
        val_transform = v2.Compose(preprocess_list)

    return train_transform, val_transform


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_train_val_test(config, preprocess_list, img2attr):
    train_transforms, val_transforms = build_transform(config, preprocess_list)

    trainset = DATASET_CLASS[config.dataset_name](
        dataset_dir=f"{config.dataset_dir}/train",
        config=config,
        img2attr=img2attr,
        transforms=train_transforms,
    )

    trainLoader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valset = DATASET_CLASS[config.dataset_name](
        dataset_dir=f"{config.dataset_dir}/val",
        config=config,
        img2attr=img2attr,
        transforms=val_transforms,
    )

    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    testset = DATASET_CLASS[config.dataset_name](
        dataset_dir=f"{config.dataset_dir}/test",
        config=config,
        img2attr=img2attr,
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


def build_optimizer(model, config):
    model_filter = filter(lambda p: p.requires_grad, model.parameters())

    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            model_filter,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd_v1":
        optimizer = optim.SGD(
            model_filter,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            model_filter,
            lr=config.lr,
            betas=(config.beta_1, config.beta_2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam_v1":
        optimizer = optim.Adam(
            model_filter,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model_filter,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    return optimizer


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


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label, concept = next(iter(loader))

    rank_zero_info(f"data shape: {data.shape}")
    rank_zero_info(f"label shape: {label.shape}")
    rank_zero_info(f"concept shape: {concept.shape}")


def print_dict(dictionary):
    text = ""
    for key, value in dictionary.items():
        text += f"{key}: {value}\n"
    rank_zero_info(text)


def step_scheduler(scheduler, config, val_loss):
    if config.scheduler in ("LinearLR", "StepLR"):
        scheduler.step()
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler.step(val_loss)


def seed_everything_in_pl():
    seed_everything(const.SEEDING, workers=True)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numerical_stability_check(cov, device, epsilon=1e-6):
    num_added = 0
    if cov.dim() == 2:
        cov = (cov + cov.transpose(dim0=0, dim1=1)) / 2
    else:
        cov = (cov + cov.transpose(dim0=1, dim1=2)) / 2

    while True:
        try:
            # Attempt Cholesky decomposition; if it fails, the matrix is not positive definite
            torch.linalg.cholesky(cov)
            if num_added > 0.0001:
                print(
                    "Added {} to the diagonal of the covariance matrix.".format(
                        num_added
                    )
                )
            break
        except RuntimeError:
            # Add epsilon to the diagonal
            if cov.dim() == 2:
                cov = cov + epsilon * torch.eye(cov.size(0), device=device)
            else:
                cov = cov + epsilon * torch.eye(cov.size(1), device=device)
            num_added += epsilon
            epsilon *= 2
    return cov


def get_empirical_covariance(dataloader):
    data = []
    for batch in dataloader:
        concepts = batch["concepts"]
        data.append(concepts)
    data = torch.cat(data)  # Concatenate all data into a single tensor
    data_logits = torch.logit(0.05 + 0.9 * data)
    covariance = torch.cov(data_logits.transpose(0, 1))

    # Bringing it into lower triangular form
    covariance = numerical_stability_check(covariance, device="cpu")
    lower_triangle = torch.linalg.cholesky(covariance)

    return lower_triangle


def load_train_state(config, model, optimizer, scaler, scheduler):
    if config.last_state is not None:
        final_model_path = f"{config.last_state}/final_model.pth"
        best_model_path = f"{config.last_state}/best_model.pth"

        print(f"Load final_model from {final_model_path}")
        ckpt = torch.load(final_model_path, map_location=const.DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if config.amp:
            scaler.load_state_dict(ckpt["scaler"])

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])

        print(f"Load best_model from {best_model_path}")
        ckpt = torch.load(best_model_path, map_location=const.DEVICE)
        best_scoring = ckpt["scoring"]
        torch.save(ckpt, f"{const.CP_PATH}/best_model.pth")
    else:
        best_scoring = 0

    return best_scoring


def train_model(train, config, scheduler, best_scoring):
    loss, metric = train.validate_one_epoch()
    print("Before training: ")
    print_dict(loss)
    print_dict(metric)

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}/{config.epochs}")

        start_epoch = time.time()

        train.epoch = epoch

        loss, metric = train.train_one_epoch()
        val_loss, val_metric = train.validate_one_epoch()

        if config.use_scheduler:
            step_scheduler(scheduler, config, val_loss["total_loss"])

        epoch_time = time.time() - start_epoch

        print(f"Epoch {epoch + 1}")
        print("Train")
        print_dict(loss)
        print_dict(metric)
        print("Val")
        print_dict(val_loss)
        print_dict(val_metric)
        print(f"epoch_time: {epoch_time} (s)\n")

        scoring = val_metric[config.best_model_criteria]
        if scoring > best_scoring:
            best_scoring = scoring
            ckpt = {
                "model": train.model.state_dict(),
                "scoring": float(best_scoring),
                "metric": metric,
                "val_metric": val_metric,
            }
            torch.save(ckpt, f"{const.CP_PATH}/best_model.pth")

    ckpt = {
        "model": train.model.state_dict(),
        "optimizer": train.optimizer.state_dict(),
        "scaler": train.scaler.state_dict() if config.amp else None,
        "scheduler": scheduler.state_dict() if config.use_scheduler else None,
    }
    torch.save(ckpt, f"{const.CP_PATH}/final_model.pth")


def get_mode(monitor):
    if monitor in const.METRIC_MAX:
        return "max"
    else:
        return "min"


def print_dist_0(text):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(text)
    else:
        print(text)


def destroy_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
