from optparse import OptionParser

from kltn_utils import kltn_utils

from ...model.cbm.train import CBMTrain
from ...model.scbm.train import MetricCalculator
from ..train_scbm import train_scbm


class CBMTrainer(train_scbm.SCBMTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.model = CBMTrain(
            CustomMetric=MetricCalculator,
            cp_path=config.cp_path,
            config=config,
            num_concept=len(self.select_concept_data["concepts"]),
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
    CBMTrainer(config).next()

    kltn_utils.rank_zero_info_newline("DONE")
