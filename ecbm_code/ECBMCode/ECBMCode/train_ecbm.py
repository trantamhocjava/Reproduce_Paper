import os
import time
from optparse import OptionParser

import torch
from torch.utils.data import BatchSampler, DataLoader

from . import utils
from .const import CLASS_NAMES, CP_PATH, DATASET_CLASS, DEVICE, GPU, SEEDING
from .dataset.dataset import ImbalancedDatasetSampler
from .loss import EBMLoss_concept, EBMLoss_label
from .model.ecbm import ECBM


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label, concept = next(iter(loader))

    print("data shape:", data.shape)
    print("label shape:", label.shape)
    print("concept shape:", concept.shape)


def load_train_val(config, preprocess_list, img2attr):
    train_transforms, val_transforms = utils.build_transform(config, preprocess_list)

    g = torch.Generator()
    g.manual_seed(SEEDING)

    trainset = DATASET_CLASS[config.dataset_name](
        f"{config.dataset_dir}/train",
        transforms=train_transforms,
        use_attr=True,
        config=config,
        img2attr=img2attr,
    )
    if config.resampling:
        sampler = BatchSampler(
            ImbalancedDatasetSampler(trainset),
            batch_size=config.batch_size,
            drop_last=True,
        )
        trainLoader = DataLoader(
            trainset,
            batch_sampler=sampler,
            num_workers=4,
            worker_init_fn=utils.seed_worker,
            generator=g,
        )
    else:
        trainLoader = DataLoader(
            trainset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            worker_init_fn=utils.seed_worker,
            generator=g,
        )

    valset = DATASET_CLASS[config.dataset_name](
        f"{config.dataset_dir}/val",
        transforms=val_transforms,
        use_attr=True,
        config=config,
        img2attr=img2attr,
    )
    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    return trainLoader, valLoader


def train_model(
    model,
    config,
    optimizer,
    scaler,
    trainLoader,
    valLoader,
    best_scoring,
    scheduler,
    loss_label,
    loss_concept,
):

    (
        loss,
        cls_loss,
        cy_loss,
        cpt_loss,
        c_overall_acc,
        c_acc,
        y_xy_acc,
        y_cy_acc,
        y_xy_bmac,
        y_cy_bmac,
    ) = utils.validation(model, valLoader, config, loss_label, loss_concept)
    print("Before training: ")
    print(
        f"loss: {loss}, cls_loss: {cls_loss}, cy_loss: {cy_loss}, cpt_loss: {cpt_loss}"
    )
    print(
        f"c_overall_acc: {c_overall_acc}, c_acc: {c_acc}, y_xy_acc: {y_xy_acc}, y_cy_acc: {y_cy_acc}, y_xy_bmac: {y_xy_bmac}, y_cy_bmac: {y_cy_bmac}"
    )

    for epoch in range(config.epochs):
        print(f"Starting epoch {epoch+1}/{config.epochs}")

        model.train()

        sum_cls_loss = 0
        sum_cy_loss = 0
        sum_cpt_loss = 0
        sum_loss = 0

        start_epoch = time.time()

        for data, label, concept in trainLoader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)
            concept = concept.long().to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if config.amp:
                # Train mixed precision
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    _, _, _, cls_loss, cy_loss, cpt_loss, loss = utils.get_loss(
                        model, data, label, concept, config, loss_label, loss_concept
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                _, _, _, cls_loss, cy_loss, cpt_loss, loss = utils.get_loss(
                    model, data, label, concept, config, loss_label, loss_concept
                )

                loss.backward()
                optimizer.step()

            sum_cls_loss += cls_loss.item()
            sum_cy_loss += cy_loss.item()
            sum_cpt_loss += cpt_loss.item()
            sum_loss += loss.item()

        train_cls_loss = sum_cls_loss / len(trainLoader)
        train_cy_loss = sum_cy_loss / len(trainLoader)
        train_cpt_loss = sum_cpt_loss / len(trainLoader)
        train_loss = sum_loss / len(trainLoader)

        (
            val_loss,
            val_cls_loss,
            val_cy_loss,
            val_cpt_loss,
            val_c_overall_acc,
            val_c_acc,
            val_y_xy_acc,
            val_y_cy_acc,
            val_y_xy_bmac,
            val_y_cy_bmac,
        ) = utils.validation(model, valLoader, config, loss_label, loss_concept)

        if config.use_scheduler:
            utils.step_scheduler(scheduler, config, val_loss)

        epoch_time = time.time() - start_epoch

        print(f"Epoch {epoch + 1}")
        print(
            f"train_cls_loss: {train_cls_loss}, train_cy_loss: {train_cy_loss}, train_cpt_loss: {train_cpt_loss}, train_loss: {train_loss}"
        )
        print(
            f"val_cls_loss: {val_cls_loss}, val_cy_loss: {val_cy_loss}, val_cpt_loss: {val_cpt_loss}, val_loss: {val_loss}"
        )
        print(
            f"val_c_overall_acc: {val_c_overall_acc}, val_c_acc: {val_c_acc}, val_y_xy_acc: {val_y_xy_acc}, val_y_cy_acc: {val_y_cy_acc}, val_y_xy_bmac: {val_y_xy_bmac}, val_y_cy_bmac: {val_y_cy_bmac}"
        )
        print(f"epoch_time: {epoch_time} (s)\n")

        scoring = val_y_xy_bmac
        if scoring > best_scoring:
            best_scoring = scoring
            ckpt = {
                "model": model.state_dict(),
                "scoring": float(best_scoring),
                "val_c_overall_acc": float(val_c_overall_acc),
                "val_c_acc": float(val_c_acc),
                "val_y_xy_acc": float(val_y_xy_acc),
                "val_y_cy_acc": float(val_y_cy_acc),
                "val_y_xy_bmac": float(val_y_xy_bmac),
                "val_y_cy_bmac": float(val_y_cy_bmac),
            }
            torch.save(ckpt, f"{CP_PATH}/best_model.pth")

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if config.amp else None,
        "scheduler": scheduler.state_dict() if config.use_scheduler else None,
    }
    torch.save(ckpt, f"{CP_PATH}/final_model.pth")


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    utils.seed_everything(SEEDING)

    os.makedirs(CP_PATH, exist_ok=True)

    config.class_names = CLASS_NAMES[config.dataset_name]

    model = ECBM(config)
    model.to(DEVICE)

    class_list = [i for i in range(len(config.class_names))]
    concept_list = [i for i in range(config.cpt_size)]
    loss_label = EBMLoss_label(class_list)
    loss_concept = EBMLoss_concept(concept_list)
    optimizer = utils.build_optimizer(model, config)
    scheduler = (
        utils.build_scheduler(optimizer, config) if config.use_scheduler else None
    )
    scaler = torch.amp.GradScaler(DEVICE) if config.amp else None

    if config.final_model is not None:
        print(f"Load final_model from {config.final_model}")
        ckpt = torch.load(config.final_model, map_location=DEVICE)
        model.load_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if config.amp:
            scaler.load_state_dict(ckpt["scaler"])

        if config.use_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])

    if config.best_model is not None:
        print(f"Load best_model from {config.best_model}")
        ckpt = torch.load(config.best_model, map_location=DEVICE)
        best_scoring = ckpt["scoring"]
        torch.save(ckpt, f"{CP_PATH}/best_model.pth")

    else:
        best_scoring = 0

    print("Load img2attr")
    img2attr = utils.load_img2attr(config)

    print("Load train, val dataset")
    trainLoader, valLoader = load_train_val(config, model.preprocess_list, img2attr)
    print("Train")
    print_shape_first_batch(trainLoader)
    print("val")
    print_shape_first_batch(valLoader)

    print("Train model")
    train_model(
        model,
        config,
        optimizer,
        scaler,
        trainLoader,
        valLoader,
        best_scoring,
        scheduler,
        loss_label,
        loss_concept,
    )

    ckpt = torch.load(f"{CP_PATH}/best_model.pth")
    print("Evaluate best_model on val")
    print(f"val_c_overall_acc: {ckpt['val_c_overall_acc']}")
    print(f"val_c_acc: {ckpt['val_c_acc']}")
    print(f"val_y_xy_acc: {ckpt['val_y_xy_acc']}")
    print(f"val_y_cy_acc: {ckpt['val_y_cy_acc']}")
    print(f"val_y_xy_bmac: {ckpt['val_y_xy_bmac']}")
    print(f"val_y_cy_bmac: {ckpt['val_y_cy_bmac']}")

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
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
        "--epochs",
        dest="epochs",
        type="int",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
    )
    parser.add_option(
        "--resampling",
        dest="resampling",
        action="store_true",
    )
    parser.add_option(
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
    )
    parser.add_option(
        "--optimizer", dest="optimizer", type="str", help="[sgd, adam, adamw]"
    )
    parser.add_option("--lr", dest="lr", type="float")
    parser.add_option(
        "--dataset_name",
        type="str",
        dest="dataset_name",
    )
    parser.add_option("--dataset_dir", type="str", dest="dataset_dir")
    parser.add_option("--amp", action="store_true", dest="amp")
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )
    parser.add_option(
        "--scheduler",
        type="str",
        dest="scheduler",
        default=None,
        help="[LinearLR, ReduceLROnPlateau]",
    )
    parser.add_option(
        "--backbone", type="str", dest="backbone", help="[resnet101_imagenet]"
    )
    parser.add_option(
        "--emb_size",
        type="int",
        dest="emb_size",
    )
    parser.add_option(
        "--hid_size",
        type="int",
        dest="hid_size",
    )
    parser.add_option(
        "--cpt_size",
        type="int",
        dest="cpt_size",
    )
    parser.add_option(
        "--freezebb",
        action="store_true",
        dest="freezebb",
    )
    parser.add_option(
        "--cpt_weight",
        type="float",
        dest="cpt_weight",
    )
    parser.add_option(
        "--cls_weight",
        type="float",
        dest="cls_weight",
    )
    parser.add_option(
        "--cy_weight",
        type="float",
        dest="cy_weight",
    )
    parser.add_option("--momentum", type="float", dest="momentum", default=None)
    parser.add_option("--beta_1", type="float", dest="beta_1", default=None)
    parser.add_option("--beta_2", type="float", dest="beta_2", default=None)
    parser.add_option("--wd", type="float", dest="wd", default=None)
    parser.add_option(
        "--cy_perturb_prob", type="float", dest="cy_perturb_prob", default=None
    )
    parser.add_option(
        "--cy_permute_prob", type="float", dest="cy_permute_prob", default=None
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
