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

    kltn_utils.rank_zero_info_newline("Load train, val dataset")
    train_transform, val_transform = kltn_utils.build_transform(config.transform)
    train_loader = train_adacbm_utils.load_dataloader(
        config, train_transform, select_concept_data["concept2class"], "train"
    )
    val_loader = train_adacbm_utils.load_dataloader(
        config, val_transform, select_concept_data["concept2class"], "val"
    )

    kltn_utils.rank_zero_info_newline("Train model")
    train.train_model(
        cp_path=config.cp_path,
        last_state=config.last_state,
        monitor=config.monitor,
        end_epoch=config.end_epoch,
        amp=config.amp,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

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
