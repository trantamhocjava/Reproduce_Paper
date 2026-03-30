import os
import time
from optparse import OptionParser

import torch
from torch.utils.data import DataLoader

from . import utils
from .const import CLASS_NAMES, DATASET_CLASS, DEVICE, GPU, SEEDING
from .model.ecbm import ECBM


def load_test(config, preprocess_list, img2attr):
    _, val_transforms = utils.build_transform(config, preprocess_list)

    g = torch.Generator()
    g.manual_seed(SEEDING)

    valset = DATASET_CLASS[config.dataset_name](
        f"{config.dataset_dir}/test",
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

    return valLoader


def load_model(config):
    model = ECBM(config)
    ckpt = torch.load(config.best_model, map_location=DEVICE)
    model.load_dict(ckpt["model"])

    return model


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    config.class_names = CLASS_NAMES[config.dataset_name]
    config.concept_group_map = utils.get_concept_group_map(config)

    print("Load img2attr")
    img2attr = utils.load_img2attr(config)

    print("Load model")
    model = load_model(config)
    model.cuda()

    print("Load test")
    test_loader = load_test(config, model.preprocess_list, img2attr)

    print("No gradient inference")
    (
        c_overall_acc,
        c_acc,
        y_xy_acc,
        y_cy_acc,
        y_xy_bmac,
        y_cy_bmac,
    ) = utils.validation_no_loss(model, test_loader)
    print(f"test_c_overall_acc: {c_overall_acc}")
    print(f"test_c_acc: {c_acc}")
    print(f"test_y_xy_acc: {y_xy_acc}")
    print(f"test_y_cy_acc: {y_cy_acc}")
    print(f"test_y_xy_bmac: {y_xy_bmac}")
    print(f"test_y_cy_bmac: {y_cy_bmac}\n")

    print("With gradient inference")
    model = load_model(config)
    model.cuda()

    start_time = time.time()
    y_acc, y_bmac, c_acc_overall, c_acc = utils.validation_gradient_infer(
        model, test_loader, config
    )
    elapse_time = time.time() - start_time
    print(f"test_c_overall_acc: {c_acc_overall}")
    print(f"test_c_acc: {c_acc}")
    print(f"test_y_acc: {y_acc}")
    print(f"test_y_bmac: {y_bmac}")
    print(f"time: {elapse_time} (s)")

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--best_model",
        type="str",
        dest="best_model",
        default=None,
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
    )
    parser.add_option(
        "--amp",
        action="store_true",
        dest="amp",
    )
    parser.add_option(
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
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
        "--patience",
        type="int",
        dest="patience",
    )
    parser.add_option(
        "--delta",
        type="float",
        dest="delta",
    )
    parser.add_option("--max_iter", type="int", dest="max_iter", default=100)
    parser.add_option(
        "--intervene_type", type="str", dest="intervene_type", default=None
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
    parser.add_option(
        "--lr_c",
        type="float",
        dest="lr_c",
    )
    parser.add_option(
        "--lr_y",
        type="float",
        dest="lr_y",
    )
    parser.add_option("--missingratio", type="float", dest="missingratio", default=None)
    parser.add_option(
        "--cy_perturb_prob", type="float", dest="cy_perturb_prob", default=None
    )
    parser.add_option(
        "--cy_permute_prob", type="float", dest="cy_permute_prob", default=None
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
