import os
from optparse import OptionParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .train import AdacbmTrain


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    rank_zero_info("Load model")
    select_concepts_data = torch.load(
        config.select_concepts_data_path, map_location="cpu"
    )

    model = AdacbmTrain(select_concepts_data=select_concepts_data, config=config)

    rank_zero_info("Load test dataset")
    _, _, testLoader = utils.load_train_val_test(config)
    rank_zero_info("test")
    utils.print_shape_first_batch(testLoader)

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    rank_zero_info("Result of best model on testset")
    tester.test(model=model, ckpt_path=config.best_model, dataloaders=testLoader)

    print("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--select_concepts_data_path",
        dest="select_concepts_data_path",
        type="str",
    )
    parser.add_option(
        "--best_model",
        type="str",
        dest="best_model",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
    )
    parser.add_option(
        "--dataset_name",
        dest="dataset_name",
        type="str",
    )
    parser.add_option(
        "--dataset_dir", type="str", dest="dataset_dir", help="the path of the dataset"
    )
    parser.add_option(
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
    )
    parser.add_option("--clip_model", dest="clip_model", type="str")
    parser.add_option("--num_layers", dest="num_layers", type="int")

    (cfg, args) = parser.parse_args()

    main(cfg)
