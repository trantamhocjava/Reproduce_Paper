from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from kltn_utils.cbm import classes as cbm_classes

from . import utils
from .model.ecbm.train import ECBMTrain, MetricCalculator


class EcbmKFoldTrainer(cbm_classes.BaseStratifiedKFoldTrainerV1):
    def __init__(self, config) -> None:
        utils.setup_train(config)
        super().__init__(config)

        self.select_concept_data = torch.load(
            config.select_concepts_data_path, map_location="cpu", weights_only=False
        )

        train_transform, val_transform = kltn_utils.build_transform(config.transform)
        self.train_dataset = utils.load_dataset(
            dataset_dir=config.dataset_dir,
            class_names=config.class_names,
            transform=train_transform,
            concept2class=self.select_concept_data["concept2class"],
            mode="train",
        )
        self.test_dataset = utils.load_dataset(
            dataset_dir=config.dataset_dir,
            class_names=config.class_names,
            transform=val_transform,
            concept2class=self.select_concept_data["concept2class"],
            mode="test",
        )

    def build_model_fn(self):
        return ECBMTrain(
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
    EcbmKFoldTrainer(config).next()

    kltn_utils.rank_zero_info_newline("DONE")
