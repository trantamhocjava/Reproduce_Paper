import os
from optparse import OptionParser

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import utils
from .const import CLASS_NAMES, CLS_WEIGHT_DICT, DEVICE
from .dataset.dataset import CustomDataset
from .model.adacbm import AdaCBM


def load_test(config, preprocess_list):
    val_transforms = v2.Compose(preprocess_list)

    valset = CustomDataset(
        dataset_dir=f"{config.dataset_dir}/test",
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

    return valLoader


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    config.class_names = CLASS_NAMES[config.dataset_name]
    config.cls_weight = CLS_WEIGHT_DICT[config.dataset_name]

    select_concepts_data = torch.load(
        config.select_concepts_save_data_path, weights_only=False
    )

    print(f"select_idx shape: {select_concepts_data['select_idx'].shape}")
    print(f"first 10 select_idx: {select_concepts_data['select_idx'][:10]}")

    model = AdaCBM(select_concepts_data=select_concepts_data, config=config)
    ckpt = torch.load(config.best_model)
    model.load_state_dict(ckpt["model"])

    criterion = utils.build_criterion(config)

    print("Load test dataset")
    test_loader = load_test(config, model.preprocess_list)

    print("Evaluate")
    model.to(DEVICE)
    bmac, acc, losses_cls = utils.validation(model, test_loader, criterion)

    print(f"test_bmac: {bmac}")
    print(f"test_acc: {acc}")
    print(f"test_losses_cls: {losses_cls}")

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--select_concepts_save_data_path",
        dest="select_concepts_save_data_path",
        type="str",
    )
    parser.add_option(
        "--clip_model",
        dest="clip_model",
        type="str",
    )
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
    parser.add_option("--batch_size", dest="batch_size", type="int")
    parser.add_option(
        "--dataset_dir", type="str", dest="dataset_dir", help="the path of the dataset"
    )
    parser.add_option(
        "--best_model", type="str", dest="best_model", help="the path of the dataset"
    )

    parser.add_option("--gpu", type="str", dest="gpu", default="0")

    (cfg, args) = parser.parse_args()
    main(cfg)
