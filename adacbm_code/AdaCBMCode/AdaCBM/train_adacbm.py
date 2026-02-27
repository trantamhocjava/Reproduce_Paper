import os
import time
from optparse import OptionParser

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import utils
from .const import CLASS_NAMES, CLS_WEIGHT_DICT, DEVICE
from .dataset.dataset import CustomDataset
from .model.adacbm import AdaCBM


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label = next(iter(loader))

    print("data shape:", data.shape)
    print("label shape:", label.shape)


def load_train_val(config, preprocess_list):
    train_transforms = v2.Compose(preprocess_list)

    val_transforms = v2.Compose(preprocess_list)

    trainset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/train",
        transforms=train_transforms,
        class_names=config.class_names,
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
        transforms=val_transforms,
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
    cls_logits, _ = model(data)

    loss_cls = criterion(cls_logits, label)

    return loss_cls


def train_model(
    model,
    config,
    optimizer,
    trainLoader,
    valLoader,
    best_scoring,
    scheduler,
    scaler,
    criterion,
):
    val_bmac, val_acc, val_losses_cls = utils.validation(model, valLoader, criterion)
    print(
        f"Before training: val_bmac: {val_bmac}, val_acc: {val_acc}, val_losses_cls: {val_losses_cls}"
    )

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}/{config.epochs}")
        epoch_loss_cls = 0

        model.train()

        start_epoch = time.time()

        for data, label in trainLoader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if config.amp:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    loss = get_loss(model, data, label, criterion)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = get_loss(model, data, label, criterion)

                loss.backward()
                optimizer.step()

            epoch_loss_cls += loss.item()

        if config.use_scheduler:
            scheduler.step()

        train_loss_cls = epoch_loss_cls / len(trainLoader)
        val_bmac, val_acc, val_losses_cls = utils.validation(
            model, valLoader, criterion
        )

        epoch_time = time.time() - start_epoch

        print(f"Epoch {epoch + 1}")
        print(f"train_loss_cls: {train_loss_cls}")
        print(
            f"val_loss_cls: {val_losses_cls}, val_BMAC: {val_bmac}, val_acc: {val_acc}"
        )
        print(f"epoch_time: {epoch_time} (s)\n")

        if val_bmac > best_scoring:
            best_scoring = val_bmac
            ckpt = {
                "model": model.state_dict(),
                "scoring": float(best_scoring),
                "val_bmac": float(val_bmac),
                "val_acc": float(val_acc),
                "val_losses_cls": float(val_losses_cls),
            }
            torch.save(ckpt, f"{config.cp_path}/best_model.pth")

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if config.use_scheduler else None,
        "scaler": scaler.state_dict() if config.amp else None,
    }
    torch.save(ckpt, f"{config.cp_path}/final_model.pth")


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


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    config.class_names = CLASS_NAMES[config.dataset_name]
    config.cls_weight = CLS_WEIGHT_DICT[config.dataset_name]

    os.makedirs(config.cp_path, exist_ok=True)

    select_concepts_data = torch.load(
        config.select_concepts_save_data_path, weights_only=False
    )

    print(f"select_idx shape: {select_concepts_data['select_idx'].shape}")
    print(f"first 10 select_idx: {select_concepts_data['select_idx'][:10]}")

    model = AdaCBM(select_concepts_data=select_concepts_data, config=config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config) if config.use_scheduler else None
    scaler = torch.amp.GradScaler(DEVICE) if config.amp else None
    criterion = utils.build_criterion(config)

    if config.final_model is not None:
        print(f"Load final_model from {config.final_model}")

        ckpt = torch.load(config.final_model, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])

        if config.amp:
            scaler.load_state_dict(ckpt["scaler"])

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
        trainLoader,
        valLoader,
        best_scoring,
        scheduler,
        scaler,
        criterion,
    )

    print("Evaluate best_model on val")
    ckpt = torch.load(f"{config.cp_path}/best_model.pth")
    print(f"val_bmac: {ckpt['val_bmac']}")
    print(f"val_acc: {ckpt['val_acc']}")
    print(f"val_losses_cls: {ckpt['val_losses_cls']}")

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--select_concepts_save_data_path",
        dest="select_concepts_save_data_path",
        type="str",
    )
    parser.add_option("--clip_model", dest="clip_model", type="str")
    parser.add_option(
        "--num_concept",
        dest="num_concept",
        type="int",
    )
    parser.add_option(
        "--dataset_name",
        dest="dataset_name",
        type="str",
    )
    parser.add_option(
        "--use_rand_init",
        dest="use_rand_init",
        action="store_true",
    )
    parser.add_option("--init_val", dest="init_val", type="float", default=1.0)
    parser.add_option(
        "--cls_sim_prior",
        dest="cls_sim_prior",
        action="store_true",
    )
    parser.add_option("--num_layers", dest="num_layers", type="int")
    parser.add_option("--residual", dest="residual", action="store_true")
    parser.add_option("--use_img_norm", dest="use_img_norm", action="store_true")
    parser.add_option("--epochs", dest="epochs", type="int")
    parser.add_option("--batch_size", dest="batch_size", type="int")
    parser.add_option("--lr", dest="lr", type="float")
    parser.add_option("--optimizer", dest="optimizer", type="str")
    parser.add_option(
        "--final_model",
        type="str",
        dest="final_model",
        default=None,
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
    )
    parser.add_option(
        "--dataset_dir", type="str", dest="dataset_dir", help="the path of the dataset"
    )
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )
    parser.add_option(
        "--amp",
        action="store_true",
        dest="amp",
    )

    parser.add_option("--gpu", type="str", dest="gpu", default="0")

    (cfg, args) = parser.parse_args()

    main(cfg)
