from optparse import OptionParser

import torch
from kltn_utils import kltn_utils, train

from ...model.adacbm.train import AdacbmTrain, MetricCalculator
from . import utils as train_adacbm_utils


def main(config):
    train_adacbm_utils.setup_train(config)

    kltn_utils.rank_zero_info_newline("Load select_concept_data")
    select_concept_data = torch.load(
        config.select_concepts_data_path, map_location="cpu", weights_only=False
    )

    kltn_utils.rank_zero_info_newline("Load model")
    model = AdacbmTrain(
        CustomMetric=MetricCalculator,
        cp_path=config.cp_path,
        config=config,
        select_concepts_data=select_concept_data,
    )

    kltn_utils.rank_zero_info_newline("Load dataset")
    _, val_transform = kltn_utils.build_transform(config.transform)
    test_loader = train_adacbm_utils.load_dataloader(
        config, val_transform, select_concept_data["concept2class"], "test"
    )

    kltn_utils.rank_zero_info_newline("Test model")
    train.test_model(model, best_model_path=config.best_model, test_loader=test_loader)

    kltn_utils.rank_zero_info_newline("Done")


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
