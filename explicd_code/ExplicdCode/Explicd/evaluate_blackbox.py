import os
from optparse import OptionParser

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import utils
from .const import CLASS_NAMES, CLS_WEIGHT_DICT, DATASET_CLASS, DEVICE


def load_test(config, model):
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)

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

    valset = DATASET_CLASS[config.dataset_name](
        f"{config.dataset_dir}/test",
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

    return valLoader


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    config.cls_weight = CLS_WEIGHT_DICT[config.dataset_name]
    config.class_names = CLASS_NAMES[config.dataset_name]

    model = utils.build_blackbox_model(config)
    ckpt = torch.load(config.best_model)
    model.load_state_dict(ckpt["model"])

    criterion = utils.build_criterion(config)

    print("Load test dataset")
    test_loader = load_test(config, model)

    print("Evaluate model")
    model.to(DEVICE)
    bmac, acc, loss_cls = utils.validation_blackbox(model, test_loader, criterion)
    print(f"test_bmac: {bmac}")
    print(f"test_acc: {acc}")
    print(f"test_loss_cls: {loss_cls}")
    print("done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=128,
        type="int",
        help="batch size",
    )
    parser.add_option(
        "--best_model",
        type="str",
        dest="best_model",
        default=None,
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

    (cfg, args) = parser.parse_args()
    main(cfg)
