from optparse import OptionParser

from kltn_utils import kltn_utils

from ...models.adacbm_hybridcbm.train import AdaHybridCBMTrain
from ...models.hybridcbm.train import MetricCalculator
from ..train_hybridcbm.train_hybridcbm_kfold import HybridcbmKFoldTrainer


class AdaHybridcbmKFoldTrainer(HybridcbmKFoldTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

    def build_model_fn(self):
        return AdaHybridCBMTrain(
            CustomMetric=MetricCalculator,
            cp_path=config.cp_path,
            config=config,
            select_concept_data=self.select_concept_data,
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
    AdaHybridcbmKFoldTrainer(config).next()

    kltn_utils.rank_zero_info_newline("DONE")
