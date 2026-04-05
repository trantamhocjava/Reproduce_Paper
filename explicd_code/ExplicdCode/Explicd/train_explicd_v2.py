import os
import time
from optparse import OptionParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import utils
from .const import (
    CLASS_NAMES,
    CLS_WEIGHT_DICT,
    CONCEPT_DATASET_DICT,
    DEVICE,
)
from .dataset.dataset import CustomDataset
from .model.explicd_v1 import ExpLICD


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label, concept_label = next(iter(loader))

    print("data shape:", data.shape)
    print("label shape:", label.shape)
    print("concept_label shape:", concept_label.shape)


def load_train_val(config, preprocess_list):
    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.75, 1.0),
                ratio=(0.75, 1.33),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ]
        + preprocess_list
    )

    val_transforms = v2.Compose(preprocess_list)

    trainset = CustomDataset(
        f"{config.dataset_dir}/train",
        transforms=train_transforms,
        return_concept_label=True,
        config=config,
    )
    trainLoader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valset = CustomDataset(
        f"{config.dataset_dir}/val",
        transforms=val_transforms,
        return_concept_label=True,
        config=config,
    )
    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    return trainLoader, valLoader


def get_loss(model, data, label, concept_label, criterion):
    cls_logits, image_logits_dict = model(data)

    loss_cls = criterion(cls_logits, label)

    loss_concepts = 0
    idx = 0
    for key in model.concept_token_dict.keys():
        image_concept_loss = F.cross_entropy(
            image_logits_dict[key], concept_label[:, idx]
        )
        loss_concepts += image_concept_loss
        idx += 1

    loss_concepts = loss_concepts / len(model.concept_token_dict.keys())
    loss = loss_cls + loss_concepts

    return loss, loss_cls, loss_concepts


def train_model(
    model,
    config,
    optimizer,
    scaler,
    trainLoader,
    valLoader,
    criterion,
    best_scoring,
    scheduler,
):

    val_bmac, val_acc, val_loss_cls, val_loss_concept = utils.validation(
        model, valLoader, criterion
    )
    print(
        f"Before training: val_bmac: {val_bmac}, val_acc: {val_acc}, val_loss_cls: {val_loss_cls}, val_loss_concept: {val_loss_concept}"
    )

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}/{config.epochs}")
        epoch_loss_cls = 0
        epoch_loss_concept = 0

        model.train()

        start_epoch = time.time()

        for data, label, concept_label in trainLoader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)
            concept_label = concept_label.long().to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if config.amp:
                # Train mixed precision
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    loss, loss_cls, loss_concepts = get_loss(
                        model, data, label, concept_label, criterion
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                loss, loss_cls, loss_concepts = get_loss(
                    model, data, label, concept_label, criterion
                )

                loss.backward()
                optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_concept += loss_concepts.item()

        if config.use_scheduler:
            scheduler.step()

        train_loss_cls = epoch_loss_cls / len(trainLoader)
        train_loss_concept = epoch_loss_concept / len(trainLoader)
        val_bmac, val_acc, val_loss_cls, val_loss_concept = utils.validation(
            model, valLoader, criterion
        )

        epoch_time = time.time() - start_epoch

        print(f"Epoch {epoch + 1}")
        print(
            f"train_loss_cls: {train_loss_cls}, train_loss_concept: {train_loss_concept}"
        )
        print(
            f"val_loss_cls: {val_loss_cls}, val_loss_concept: {val_loss_concept}, val_BMAC: {val_bmac}, val_acc: {val_acc}"
        )
        print(f"epoch_time: {epoch_time} (s)\n")

        if val_bmac > best_scoring:
            best_scoring = val_bmac
            ckpt = {
                "model": model.state_dict(),
                "scoring": float(best_scoring),
                "val_bmac": float(val_bmac),
                "val_acc": float(val_acc),
                "val_loss_cls": float(val_loss_cls),
                "val_loss_concept": float(val_loss_concept),
            }
            torch.save(ckpt, f"{config.cp_path}/best_model.pth")

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if config.amp else None,
        "scheduler": scheduler.state_dict() if config.use_scheduler else None,
    }
    torch.save(ckpt, f"{config.cp_path}/final_model.pth")


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    print(f"clip_model: {config.clip_model}")

    os.makedirs(config.cp_path, exist_ok=True)

    config.cls_weight = CLS_WEIGHT_DICT[config.dataset_name]
    config.class_names = CLASS_NAMES[config.dataset_name]

    model = ExpLICD(
        concept_list=CONCEPT_DATASET_DICT[config.dataset_name],
        clip_model=config.clip_model,
        config=config,
    )
    criterion = utils.build_criterion(config)
    optimizer = utils.build_optimizer(model, config)
    scheduler = (
        utils.build_scheduler(optimizer, config) if config.use_scheduler else None
    )
    scaler = torch.amp.GradScaler(DEVICE) if config.amp else None

    if config.final_model is not None:
        print(f"Load final_model from {config.final_model}")
        ckpt = torch.load(config.final_model, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if config.amp:
            scaler.load_state_dict(ckpt["scaler"])

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])

    if config.best_model is not None:
        print(f"Load best_model from {config.best_model}")
        ckpt = torch.load(
            config.best_model,
        )
        best_scoring = ckpt["scoring"]
        torch.save(ckpt, f"{config.cp_path}/best_model.pth")

    else:
        best_scoring = 0

    print("Load train, val dataset")
    trainLoader, valLoader = load_train_val(config, model.preprocess_list)
    print("Train")
    print_shape_first_batch(trainLoader)
    print("val")
    print_shape_first_batch(valLoader)

    print("Train model")
    model.to(DEVICE)
    train_model(
        model,
        config,
        optimizer,
        scaler,
        trainLoader,
        valLoader,
        criterion,
        best_scoring,
        scheduler,
    )

    print("Evaluate best_model on val")
    ckpt = torch.load(f"{config.cp_path}/best_model.pth")
    print(f"val_bmac: {ckpt['val_bmac']}")
    print(f"val_acc: {ckpt['val_acc']}")
    print(f"val_loss_cls: {ckpt['val_loss_cls']}")
    print(f"val_loss_concept: {ckpt['val_loss_concept']}")

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--epochs",
        dest="epochs",
        default=150,
        type="int",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=128,
        type="int",
    )
    parser.add_option("--optimizer", dest="optimizer", default="adamw", type="str")
    parser.add_option(
        "--lr", dest="lr", default=0.0001, type="float", help="learning rate"
    )
    parser.add_option(
        "--final_model",
        type="str",
        dest="final_model",
        default=None,
    )
    parser.add_option(
        "-p",
        "--checkpoint-path",
        type="str",
        dest="cp_path",
        default="/kaggle/working/checkpoint",
        help="checkpoint path",
    )
    parser.add_option(
        "--dataset_name",
        type="str",
        dest="dataset_name",
    )
    parser.add_option(
        "--dataset_dir", type="str", dest="dataset_dir", help="the path of the dataset"
    )
    parser.add_option(
        "--amp", action="store_true", dest="amp", help="if use mixed precision training"
    )
    parser.add_option(
        "--best_model",
        type="str",
        dest="best_model",
        default=None,
    )
    parser.add_option(
        "--clip_model",
        type="str",
        dest="clip_model",
    )
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )
    parser.add_option("--gpu", type="str", dest="gpu", default="0")

    (cfg, args) = parser.parse_args()

    main(cfg)
