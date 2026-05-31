from optparse import OptionParser

import torch
from kltn_utils import kltn_class, kltn_utils

from ...model.adacbm.train import AdacbmTrain, MetricCalculator
from . import utils as train_adacbm_utils


class AdaCBMTrainer(kltn_class.BaseTrainer):
    def __init__(self, config) -> None:
        train_adacbm_utils.setup_train(config)
        super().__init__(config)

        select_concept_data = torch.load(
            config.select_concepts_data_path, map_location="cpu", weights_only=False
        )

        self.model = AdacbmTrain(
            CustomMetric=MetricCalculator,
            cp_path=config.cp_path,
            config=config,
            select_concepts_data=select_concept_data,
        )

        train_transform, val_transform = kltn_utils.build_transform(config.transform)
        self.train_loader = train_adacbm_utils.load_dataloader(
            config, train_transform, select_concept_data["concept2class"], "train"
        )
        self.val_loader = train_adacbm_utils.load_dataloader(
            config, val_transform, select_concept_data["concept2class"], "val"
        )
        self.test_loader = train_adacbm_utils.load_dataloader(
            config, val_transform, select_concept_data["concept2class"], "test"
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
    AdaCBMTrainer(config).next()

    kltn_utils.rank_zero_info_newline("DONE")
