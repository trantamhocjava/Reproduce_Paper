import os
from optparse import OptionParser

from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .train import JointCBMTrainPL


def main(config):
    # os.environ["CUDA_VISIBLE_DEVICES"] = const.GPU
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    utils.seed_everything_in_pl()

    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    rank_zero_info("Load model")
    model = JointCBMTrainPL(config=config)

    rank_zero_info("Load test dataset")
    img2attr = utils.load_img2attr(config)

    _, _, testLoader = utils.load_train_val_test(
        config, model.model.preprocess_list, img2attr
    )
    rank_zero_info("test")
    utils.print_shape_first_batch(testLoader)

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    rank_zero_info("Result of best model on testset")
    tester.test(model=model, ckpt_path=config.best_model, dataloaders=testLoader)

    utils.destroy_process_group()

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
        "--epochs",
        dest="epochs",
        type="int",
    )
    parser.add_option(
        "--training_mode",
        dest="training_mode",
        type="str",
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
    parser.add_option("--dataset_dir", type="str", dest="dataset_dir")
    parser.add_option(
        "--num_concepts",
        type="int",
        dest="num_concepts",
    )
    parser.add_option(
        "--head_arch", type="str", dest="head_arch", help="[linear, nonlinear]"
    )
    parser.add_option("--alpha", type="float", dest="alpha", default=None)
    parser.add_option(
        "--encoder_arch",
        type="str",
        dest="encoder_arch",
        help="[resnet18, simple_CNN, FCNN]",
    )
    parser.add_option(
        "--decrease_every",
        type="int",
        dest="decrease_every",
    )
    parser.add_option(
        "--lr_divisor",
        type="int",
        dest="lr_divisor",
    )
    parser.add_option(
        "--weight_decay",
        type="float",
        dest="weight_decay",
    )
    parser.add_option(
        "--compile",
        action="store_true",
        dest="compile",
    )
    parser.add_option(
        "--num_monte_carlo",
        type="int",
        dest="num_monte_carlo",
    )
    parser.add_option(
        "--straight_through",
        action="store_true",
        dest="straight_through",
    )
    parser.add_option(
        "--concept_learning", type="str", dest="concept_learning", default=None
    )
    parser.add_option("--inter_policy", type="str", dest="inter_policy", default=None)
    parser.add_option(
        "--inter_strategy", type="str", dest="inter_strategy", default=None
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
