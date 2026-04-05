import os
import time
from optparse import OptionParser

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import utils
from .const import CLASS_NAMES, CLS_WEIGHT_DICT, DATASET_CLASS, DEVICE


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label = next(iter(loader))

    print("data shape:", data.shape)
    print("label shape:", label.shape)


def load_train_val(config, model):
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)

    transform_list = [
        v2.RandomResizedCrop(
            size=data_cfg["input_size"][-1],
            scale=(0.75, 1.0),
            ratio=(0.75, 1.33),
            interpolation=utils.get_interpolation_mode(data_cfg["interpolation"]),
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
    ]

    if config.dataset_name == "isic2018":
        transform_list.insert(4, utils.GrayWorld())

    train_transforms = v2.Compose(transform_list)

    transform_list = [
        v2.Resize(
            size=int(data_cfg["input_size"][-1] / data_cfg["crop_pct"]),
            interpolation=utils.get_interpolation_mode(data_cfg["interpolation"]),
        ),
        v2.CenterCrop(size=data_cfg["input_size"][-1]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
    ]

    if config.dataset_name == "isic2018":
        transform_list.insert(3, utils.GrayWorld())

    val_transforms = v2.Compose(transform_list)

    trainset = DATASET_CLASS[config.dataset_name](
        f"{config.dataset_dir}/train",
        transforms=train_transforms,
        return_concept_label=False,
        class_names=config.class_names,
    )
    trainLoader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valset = DATASET_CLASS[config.dataset_name](
        f"{config.dataset_dir}/val",
        transforms=val_transforms,
        return_concept_label=False,
        class_names=config.class_names,
    )
    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    return trainLoader, valLoader


def get_loss(model, data, label, criterion):
    output = model(data)

    loss = criterion(output, label)

    return loss


def train_model(
    model,
    config,
    trainLoader,
    valLoader,
    criterion,
    optimizer,
    scaler,
    best_scoring,
    scheduler,
):
    bmac, acc, loss_cls = utils.validation_blackbox(model, valLoader, criterion)
    print(
        f"Before training: val_bmac: {bmac}, val_acc: {acc}, val_loss_cls: {loss_cls}"
    )

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}/{config.epochs}")
        epoch_loss = 0

        model.train()

        start_epoch = time.time()

        for data, label in trainLoader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if config.amp:
                with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                    loss = get_loss(model, data, label, criterion)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = get_loss(model, data, label, criterion)

                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        if config.use_scheduler:
            scheduler.step()

        
        train_loss_cls = epoch_loss / len(trainLoader)
        val_bmac, val_acc, val_loss_cls = utils.validation_blackbox(
            model, valLoader, criterion
        )

        epoch_time = time.time() - start_epoch

        print(f"Epoch {epoch + 1}")
        print(f"train_loss_cls: {train_loss_cls}")
        print(
            f"val_loss_cls: {val_loss_cls},  val_BMAC: {val_bmac}, val_acc: {val_acc}"
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

    os.makedirs(config.cp_path, exist_ok=True)

    config.cls_weight = CLS_WEIGHT_DICT[config.dataset_name]
    config.class_names = CLASS_NAMES[config.dataset_name]

    model = utils.build_blackbox_model(config)
    optimizer = utils.build_optimizer(config, model)
    criterion = utils.build_criterion(config)
    scaler = torch.amp.GradScaler(DEVICE) if config.amp else None
    scheduler = (
        utils.build_scheduler(optimizer, config) if config.use_scheduler else None
    )

    if config.final_model:
        print(f"Load final_model from {config.final_model}")
        ckpt = torch.load(config.final_model, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if config.amp:
            scaler.load_state_dict(ckpt["scaler"])

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])

    if config.best_model:
        print(f"Load best_model from {config.best_model}")
        ckpt = torch.load(
            config.best_model,
        )
        best_scoring = ckpt["scoring"]
        torch.save(ckpt, f"{config.cp_path}/best_model.pth")

    else:
        best_scoring = 0

    print("Load train, val, test dataset")
    trainLoader, valLoader = load_train_val(config, model)
    print("Train")
    print_shape_first_batch(trainLoader)
    print("val")
    print_shape_first_batch(valLoader)

    print("Train model")
    model.to(DEVICE)
    train_model(
        model,
        config,
        trainLoader,
        valLoader,
        criterion,
        optimizer,
        scaler,
        best_scoring,
        scheduler,
    )

    print("Evaluate best_model on val")
    ckpt = torch.load(f"{config.cp_path}/best_model.pth")
    print(f"val_bmac: {ckpt['val_bmac']}")
    print(f"val_acc: {ckpt['val_acc']}")
    print(f"val_loss_cls: {ckpt['val_loss_cls']}")

    print("done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--epochs",
        dest="epochs",
        default=150,
        type="int",
        help="number of epochs",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=128,
        type="int",
        help="batch size",
    )
    parser.add_option("--optimizer", dest="optimizer", default="sgd", type="str")
    parser.add_option(
        "--lr", dest="lr", default=0.01, type="float", help="learning rate"
    )
    parser.add_option(
        "--final_model",
        type="str",
        dest="final_model",
        default=None,
        help="load pretrained model",
    )
    parser.add_option(
        "--best_model",
        type="str",
        dest="best_model",
        default=None,
    )
    parser.add_option(
        "--checkpoint-path",
        type="str",
        dest="cp_path",
        default="/kaggle/working/checkpoint",
        help="checkpoint path",
    )
    parser.add_option(
        "--model",
        type="str",
        dest="model",
        default="resnet50.a1_in1k",
        help="use which model in [vit_base_patch16_224.orig_in21k, resnet50.a1_in1k]",
    )  # We find vit.orig_in21k is better than CLIP weights
    parser.add_option(
        "--linear-probe",
        dest="linear_probe",
        action="store_true",
        help="if use linear probe finetuning",
    )
    parser.add_option(
        "--dataset_name",
        type="str",
        dest="dataset_name",
        default="isic2018",
        help="name of datasets",
    )
    parser.add_option(
        "--dataset_dir",
        type="str",
        dest="dataset_dir",
        help="the path of the dataset",
    )
    parser.add_option("--gpu", type="str", dest="gpu", default="0")
    parser.add_option(
        "--amp", action="store_true", dest="amp", help="if use mixed precision training"
    )
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
