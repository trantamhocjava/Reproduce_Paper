import copy
import os
import time
from optparse import OptionParser

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from concept_dataset import explicid_isic_dict
from model.explicd import ExpLICD
from sklearn.metrics import balanced_accuracy_score
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ExplicdCode.Explicd.dataset.dataset import SkinDataset

DEBUG = False

dataset_dict = {"isic2018": SkinDataset}

num_class_dict = {
    "isic2018": 7,
}

cls_weight_dict = {
    "isic2018": [1, 0.5, 1.2, 1.3, 1, 2, 2],
}


def print_shape_first_batch(loader):
    # Lấy batch đầu tiên
    data, label, concept_label = next(iter(loader))

    print("data shape:", data.shape)
    print("label shape:", label.shape)
    print("concept_label shape:", concept_label.shape)


def load_train_val_test(config):
    train_transforms = copy.deepcopy(config.preprocess)
    train_transforms.transforms.pop(0)

    if model.model_name != "clip":
        train_transforms.transforms.pop(0)

    train_transforms.transforms.insert(0, transforms.RandomVerticalFlip())
    train_transforms.transforms.insert(0, transforms.RandomHorizontalFlip())
    train_transforms.transforms.insert(
        0,
        transforms.RandomResizedCrop(
            size=(224, 224),
            scale=(0.75, 1.0),
            ratio=(0.75, 1.33),
            interpolation=utils.get_interpolation_mode("bicubic"),
        ),
    )
    train_transforms.transforms.insert(0, transforms.ToPILImage())
    # if config.dataset == 'isic2018':
    #    train_transforms.transforms.insert(-1, utils.gray_world())

    val_transforms = copy.deepcopy(config.preprocess)
    val_transforms.transforms.insert(0, transforms.ToPILImage())
    # if config.dataset == 'isic2018':
    #    val_transforms.transforms.insert(-1, utils.gray_world())

    trainset = dataset_dict[config.dataset](
        config.data_path,
        mode="train",
        transforms=train_transforms,
        flag=config.flag,
        debug=DEBUG,
        config=config,
        return_concept_label=True,
    )
    trainLoader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    valset = dataset_dict[config.dataset](
        config.data_path,
        mode="val",
        transforms=val_transforms,
        flag=config.flag,
        debug=DEBUG,
        config=config,
        return_concept_label=True,
    )
    valLoader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    testset = dataset_dict[config.dataset](
        config.data_path,
        mode="test",
        transforms=val_transforms,
        flag=config.flag,
        debug=DEBUG,
        config=config,
        return_concept_label=True,
    )
    testLoader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    return trainLoader, valLoader, testLoader


def train_net(
    model, config, optimizer, scaler, trainLoader, valLoader, criterion, best_BMAC
):
    writer = SummaryWriter(config.log_path)

    BMAC, acc, _, _ = validation(model, valLoader, criterion)
    print("Pretrained model: BMAC: %.5f, Acc: %.5f" % (BMAC, acc))

    for epoch in range(config.epochs):
        print("Starting epoch {}/{}".format(epoch + 1, config.epochs))
        epoch_loss_cls = 0
        epoch_loss_concept = 0

        model.train()

        start_epoch = time.time()

        for i, (data, label, concept_label) in enumerate(trainLoader, 0):
            start_batch = time.time()
            x, target = data.float().cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()

            optimizer.zero_grad()

            if config.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    cls_logits, image_logits_dict = model(x)

                    loss_cls = criterion(cls_logits, target)

                    loss_concepts = 0
                    idx = 0
                    for key in model.concept_token_dict.keys():
                        image_concept_loss = F.cross_entropy(
                            image_logits_dict[key], concept_label[:, idx]
                        )
                        loss_concepts += image_concept_loss
                        idx += 1

                    loss = loss_cls + loss_concepts / idx

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                cls_logits, image_logits_dict = model(x)

                loss_cls = criterion(cls_logits, target)

                loss_concepts = 0
                idx = 0
                for key in model.concept_token_dict.keys():
                    image_concept_loss = F.cross_entropy(
                        image_logits_dict[key], concept_label[:, idx]
                    )
                    loss_concepts += image_concept_loss
                    idx += 1

                loss = loss_cls + loss_concepts / idx

                loss.backward()
                optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_concept += loss_concepts.item()

            batch_time = time.time() - start_batch

        epoch_time = time.time() - start_epoch

        print(
            f"[epoch {epoch + 1}] epoch_loss_cls: {epoch_loss_cls / (i + 1)}, epoch_loss_concept: {epoch_loss_concept / (i + 1)}, epoch_time: {epoch_time}"
        )

        writer.add_scalar("Train/Loss_cls", epoch_loss_cls / (i + 1), epoch + 1)
        writer.add_scalar("Train/Loss_concept", epoch_loss_concept / (i + 1), epoch + 1)

        if (epoch + 1) % 50 == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            }
            torch.save(ckpt, f"{config.cp_path}/CP{epoch + 1}.pth")

        val_BMAC, val_acc, val_loss_cls, val_loss_concept = validation(
            model, valLoader, criterion
        )
        writer.add_scalar("Val/BMAC", val_BMAC, epoch + 1)
        writer.add_scalar("Val/Acc", val_acc, epoch + 1)
        writer.add_scalar("Val/val_loss_cls", val_loss_cls, epoch + 1)
        writer.add_scalar("Val/val_loss_concept", val_loss_concept, epoch + 1)

        lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR/lr", lr, epoch + 1)

        if val_BMAC > best_BMAC:
            best_BMAC = val_BMAC
            ckpt = {
                "model": model.state_dict(),
                "BMAC": float(best_BMAC),
            }
            torch.save(ckpt, f"{config.cp_path}/best_model.pth")

        print("save done")

    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(ckpt, f"{config.cp_path}/final_model.pth")


def validation(model, dataloader, criterion):
    net = model

    net.eval()

    losses_cls = 0
    losses_concepts = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    with torch.no_grad():
        for i, (data, label, concept_label) in enumerate(dataloader):

            data, label = data.cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()
            cls_logits, image_logits_dict = net(data)

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

            losses_concepts += tmp_loss_concepts / len(
                list(model.concept_token_dict.keys())
            )

            _, label_pred = torch.max(cls_logits, dim=1)

            pred_list = np.concatenate(
                (pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_list = np.concatenate(
                (gt_list, label.cpu().numpy().astype(np.uint8)), axis=0
            )

    BMAC = balanced_accuracy_score(gt_list, pred_list) * 100
    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)

    return BMAC, acc, losses_cls / (i + 1), losses_concepts / (i + 1)


def build_model(config):
    model = ExpLICD(
        concept_list=explicid_isic_dict, model_name="biomedclip", config=config
    )

    # We find using orig_in21k vit weights works better than biomedclip vit weights
    # Delete the following if want to use biomedclip weights
    vit = timm.create_model(
        "vit_base_patch16_224.orig_in21k", pretrained=True, num_classes=config.num_class
    )
    vit.head = nn.Identity()
    model.model.visual.trunk.load_state_dict(vit.state_dict())

    return model


def toDeviceOptimizer(optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--epochs",
        dest="epochs",
        default=150,
        type="int",
        help="number of epochs",
    )
    parser.add_option(
        "-b",
        "--batch_size",
        dest="batch_size",
        default=128,
        type="int",
        help="batch size",
    )
    parser.add_option("--warmup_epoch", dest="warmup_epoch", default=5, type="int")
    parser.add_option("--optimizer", dest="optimizer", default="adamw", type="str")
    parser.add_option(
        "-l", "--lr", dest="lr", default=0.0001, type="float", help="learning rate"
    )
    parser.add_option(
        "-c",
        "--resume",
        type="str",
        dest="resume",
        default=False,
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
        "-o",
        "--log-path",
        type="str",
        dest="log_path",
        default="/kaggle/working/log",
        help="log path",
    )
    parser.add_option(
        "--linear-probe",
        dest="linear_probe",
        action="store_true",
        help="if use linear probe finetuning",
    )
    parser.add_option(
        "-d",
        "--dataset",
        type="str",
        dest="dataset",
        default="isic2018",
        help="name of dataset",
    )
    parser.add_option(
        "--data-path", type="str", dest="data_path", help="the path of the dataset"
    )
    parser.add_option(
        "-u",
        "--unique_name",
        type="str",
        dest="unique_name",
        default="test",
        help="name prefix",
    )

    parser.add_option("--flag", type="int", dest="flag", default=2)

    parser.add_option("--gpu", type="str", dest="gpu", default="0")
    parser.add_option(
        "--amp", action="store_true", dest="amp", help="if use mixed precision training"
    )
    parser.add_option(
        "--bestModel",
        type="str",
        dest="bestModel",
        default=False,
    )

    (config, args) = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    config.log_path = f"{config.log_path}/{config.dataset}"
    config.cp_path = f"{config.cp_path}/{config.dataset}"

    os.makedirs(config.cp_path, exist_ok=True)

    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]

    model = build_model(config)

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

    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    if config.resume:
        ckpt = torch.load(
            config.resume,
        )
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        toDeviceOptimizer(optimizer)

        if config.amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

        print(f"Load final_model from {config.resume}")

    if config.bestModel:
        ckpt = torch.load(
            config.bestModel,
        )
        best_BMAC = ckpt["BMAC"]
        torch.save(ckpt, f"{config.cp_path}/best_model.pth")
        print(f"Load best_model from {config.bestModel}")
    else:
        best_BMAC = 0

    if config.cls_weight == None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        lesion_weight = torch.FloatTensor(config.cls_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).cuda()

    print("Load train, val, test dataset")
    trainLoader, valLoader, testLoader = load_train_val_test(config)
    print("Train")
    print_shape_first_batch(trainLoader)
    print("val")
    print_shape_first_batch(valLoader)
    print("test")
    print_shape_first_batch(testLoader)

    print("Train model")
    model.cuda()
    train_net(
        model, config, optimizer, scaler, trainLoader, valLoader, criterion, best_BMAC
    )

    print("Evaluate on test")
    best_model = build_model(config)
    ckpt = torch.load(f"{config.cp_path}/best_model.pth")
    best_model.load_state_dict(ckpt["model"])
    best_model.cuda()

    test_BMAC, test_acc, test_loss_cls, test_loss_concept = validation(
        model, testLoader, criterion
    )
    print(f"test_BMAC: {test_BMAC}")
    print(f"test_acc: {test_acc}")
    print(f"test_loss_cls: {test_loss_cls}")
    print(f"test_loss_concept: {test_loss_concept}")

    print("done")
