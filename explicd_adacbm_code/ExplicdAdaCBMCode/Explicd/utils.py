import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import optim
from torchvision.io import ImageReadMode, read_image

from .const import DEVICE


class GrayWorld(object):
    def __call__(self, img):
        mu_g = img[1].mean()
        img[0] = img[0] * (mu_g / img[0].mean())
        img[2] = img[2] * (mu_g / img[2].mean())

        img = torch.clamp(img, 0, 1)

        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


def get_interpolation_mode(interpolation_str):
    interpolation_mapping = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "lanczos": transforms.InterpolationMode.LANCZOS,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "box": transforms.InterpolationMode.BOX,
        "hamming": transforms.InterpolationMode.HAMMING,
    }
    return interpolation_mapping.get(interpolation_str)


def validation(model, dataloader, criterion):
    model.eval()

    losses_cls = 0
    losses_concepts = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    with torch.no_grad():
        for data, label, concept_label in dataloader:
            data, label = data.float().cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()
            cls_logits, image_logits_dict = model(data)

            loss_cls = criterion(cls_logits, label)
            losses_cls += loss_cls.item()

            tmp_loss_concepts = 0
            idx = 0
            for key in model.concept_token_dict.keys():
                image_concept_loss = F.cross_entropy(
                    image_logits_dict[key], concept_label[:, idx]
                )
                tmp_loss_concepts += image_concept_loss.item()
                idx += 1

            losses_concepts += tmp_loss_concepts / len(model.concept_token_dict.keys())

            _, label_pred = torch.max(cls_logits, dim=1)

            pred_list = np.concatenate(
                (pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_list = np.concatenate(
                (gt_list, label.cpu().numpy().astype(np.uint8)), axis=0
            )

    BMAC = balanced_accuracy_score(gt_list, pred_list) * 100
    acc = accuracy_score(gt_list, pred_list) * 100
    losses_cls = losses_cls / len(dataloader)
    losses_concepts = losses_concepts / len(dataloader)

    return BMAC, acc, losses_cls, losses_concepts


def build_criterion(config):
    if config.cls_weight is None:
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    else:
        lesion_weight = torch.FloatTensor(config.cls_weight).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).to(DEVICE)

    return criterion


def validation_blackbox(model, dataloader, criterion):
    losses = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)

            pred = model(data)

            loss = criterion(pred, label)
            losses += loss.item()

            _, label_pred = torch.max(pred, dim=1)

            pred_list = np.concatenate(
                (pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_list = np.concatenate(
                (gt_list, label.cpu().numpy().astype(np.uint8)), axis=0
            )

    loss_cls = losses / len(dataloader)
    bmac = balanced_accuracy_score(gt_list, pred_list) * 100
    acc = accuracy_score(gt_list, pred_list) * 100

    return bmac, acc, loss_cls


def build_blackbox_model(config):
    print(f"use model {config.model}")

    model = timm.create_model(
        config.model, pretrained=True, num_classes=len(config.class_names)
    )
    if config.linear_probe:
        for name, param in model.named_parameters():
            if "fc" in name and "resnet" in config.model:
                param.requires_grad = True
            elif "head" in name and "vit" in config.model:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model


def build_optimizer(model, config):
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            [
                {"params": model.get_backbone_params(), "lr": config.lr * 0.1},
                {"params": model.get_bridge_params(), "lr": config.lr},
            ]
        )
    elif config.optimizer == "adamw_v1":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    return optimizer


def build_scheduler(optimizer, config):
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.01,
        total_iters=config.epochs,
    )
    return scheduler


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
