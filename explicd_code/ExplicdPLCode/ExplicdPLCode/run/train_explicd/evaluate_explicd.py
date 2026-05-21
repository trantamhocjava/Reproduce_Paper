from optparse import OptionParser

from kltn_utils import kltn_utils, train

from ...model.explicd_dict.train import ExplicdTrain, MetricCalculator
from . import utils as train_explicd_utils


def main(config):
    train_explicd_utils.setup_train(config)

    kltn_utils.rank_zero_info_newline("LOAD MODEL")
    model = ExplicdTrain(
        CustomMetric=MetricCalculator, cp_path=config.cp_path, config=config
    )

    kltn_utils.rank_zero_info_newline("LOAD TEST DATASET")
    _, val_transform = kltn_utils.build_transform(config.transform)
    test_loader = train_explicd_utils.load_dataset(config, val_transform, "test")

    kltn_utils.rank_zero_info_newline("Train model")
    train.test_model(
        model=model, best_model_path=config.best_model, test_loader=test_loader
    )

    kltn_utils.rank_zero_info_newline("DONE")


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
