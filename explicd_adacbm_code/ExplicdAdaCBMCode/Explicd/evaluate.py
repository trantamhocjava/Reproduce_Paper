import os
from optparse import OptionParser

import torch
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
from .model.explicd_adacbm import ExpLICDAdaCBM


def load_test(config, preprocess_list):
    val_transforms = v2.Compose(preprocess_list)

    valset = CustomDataset(
        f"{config.dataset_dir}/test",
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

    return valLoader


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    config.cls_weight = CLS_WEIGHT_DICT[config.dataset_name]
    config.class_names = CLASS_NAMES[config.dataset_name]

    model = ExpLICDAdaCBM(
        concept_list=CONCEPT_DATASET_DICT[config.dataset_name],
        clip_model=config.clip_model,
        config=config,
    )
    ckpt = torch.load(config.best_model)
    model.load_state_dict(ckpt["model"])

    criterion = utils.build_criterion(config)

    print("Load test dataset")
    test_loader = load_test(config, model.preprocess_list)

    print("evaluate model")
    model.to(DEVICE)
    bmac, acc, losses_cls, losses_concepts = utils.validation(
        model, test_loader, criterion
    )
    print(f"test_bmac: {bmac}")
    print(f"test_acc: {acc}")
    print(f"test_losses_cls: {losses_cls}")
    print(f"test_losses_concepts: {losses_concepts}")
    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        default=128,
        type="int",
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
        "--num_layers",
        type="int",
        dest="num_layers",
    )
    parser.add_option(
        "--residual",
        action="store_true",
        dest="residual",
    )
    parser.add_option(
        "--use_img_norm",
        action="store_true",
        dest="use_img_norm",
    )
    parser.add_option("--gpu", type="str", dest="gpu", default="0")

    (cfg, args) = parser.parse_args()
    main(cfg)
