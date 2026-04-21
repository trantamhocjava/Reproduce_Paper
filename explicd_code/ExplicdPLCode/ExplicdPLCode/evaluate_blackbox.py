import os
from optparse import OptionParser

from kltn_utils import kltn_utils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .train import BlackBoxTrain


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    kltn_utils.seed_everything_in_pl()

    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    rank_zero_info("Load model")
    model = BlackBoxTrain(config=config)

    rank_zero_info("Load test dataset")
    _, _, testLoader = utils.load_train_val_test_for_blackbox(config)
    rank_zero_info("test")
    utils.print_shape_first_batch(testLoader)

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    rank_zero_info("Result of best model on testset")
    tester.test(model=model, ckpt_path=config.best_model, dataloaders=testLoader)

    kltn_utils.destroy_process_group()

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
        "--model",
        type="str",
        dest="model",
        help="[vit_base_patch16_224.orig_in21k, resnet50.a1_in1k]",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
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

    (cfg, args) = parser.parse_args()

    main(cfg)
