from optparse import OptionParser

from kltn_utils import kltn_utils, train
from pytorch_lightning.utilities import rank_zero_info

from ...model.blackbox.train import BlackboxTrain, CustomMetric
from ..train_explicd import utils as train_explicd_utils
from . import utils as train_blackbox_utils


def main(config):
    train_explicd_utils.setup_train(config)

    rank_zero_info("Load model")
    model = BlackboxTrain(
        CustomMetric=CustomMetric, cp_path=config.cp_path, config=config
    )

    rank_zero_info("Load train, val dataset")
    _, val_transform = kltn_utils.build_transform(config.transform_method)
    test_loader = train_blackbox_utils.load_dataset(config, val_transform, "test")

    kltn_utils.rank_zero_info_newline("Evaluate model")
    train.test_model(
        model=model, best_model_path=config.best_model, test_loader=test_loader
    )

    rank_zero_info("Done")


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    cfg, args = parser.parse_args()
    cfg = kltn_utils.read_json_to_namespace(cfg.arg_json)

    main(cfg)
