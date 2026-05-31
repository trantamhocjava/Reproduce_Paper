from optparse import OptionParser

from kltn_utils import kltn_class, kltn_utils

from ...model.explicd_dict.train import ExplicdTrain, MetricCalculator
from . import utils as train_explicd_utils


class ExplicdTrainer(kltn_class.BaseTrainer):
    def __init__(self, config) -> None:
        train_explicd_utils.setup_train(config)
        super().__init__(config)

        self.model = ExplicdTrain(
            CustomMetric=MetricCalculator, cp_path=config.cp_path, config=config
        )

        train_transform, val_transform = kltn_utils.build_transform(config.transform)
        self.train_loader = train_explicd_utils.load_dataset(
            config, train_transform, "train"
        )
        self.val_loader = train_explicd_utils.load_dataset(config, val_transform, "val")
        self.test_loader = train_explicd_utils.load_dataset(
            config, val_transform, "test"
        )


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    config, args = parser.parse_args()
    config = kltn_utils.read_json_to_namespace(config.arg_json)
    ExplicdTrainer(config).next()

    kltn_utils.rank_zero_info_newline("DONE")
