import os
from optparse import OptionParser

from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .intervene.intervene_cbm import intervene_cbm
from .model.cbm.cbm import CBM


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    utils.seed_everything_in_pl()

    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    rank_zero_info("Load model")
    model = CBM(config=config)

    rank_zero_info("Load train, test dataset")
    img2attr = utils.load_img2attr(config)

    _, _, test_loader = utils.load_train_val_test(
        config, model.preprocess_list, img2attr
    )
    rank_zero_info("Test")
    utils.print_shape_first_batch(test_loader)

    rank_zero_info("Intervene")
    utils.create_csv_file(f"{const.CP_PATH}/result.csv", const.COLUMNS)
    intervene_cbm(test_loader, config)

    utils.destroy_process_group()

    rank_zero_info("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--best_model",
        type="str",
        dest="best_model",
        default=None,
    )
    parser.add_option(
        "--min_num_interventions",
        type="int",
        dest="min_num_interventions",
    )
    parser.add_option(
        "--start_epoch",
        dest="start_epoch",
        type="int",
    )
    parser.add_option(
        "--end_epoch",
        dest="end_epoch",
        type="int",
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
        "--freezebb",
        action="store_true",
        dest="freezebb",
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

    cfg, args = parser.parse_args()

    main(cfg)
