import os
from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from pytorch_lightning import Trainer

from ... import const
from ...models.hybridcbm.train import HybridCBMTrain
from . import utils as train_hybridcbm_utils


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config = train_hybridcbm_utils.update_config_for_train(config)

    kltn_utils.rank_zero_info_newline("Load select concept data")
    select_concept_data = torch.load(
        config.select_concept_data_path, map_location="cpu", weights_only=False
    )

    kltn_utils.rank_zero_info_newline("Load test dataset")
    _, val_transform = kltn_utils.build_transform(config.transform_method)
    test_loader = train_hybridcbm_utils.load_dataloader(
        config, val_transform, select_concept_data["concept2class"], "test"
    )

    kltn_utils.rank_zero_info_newline("Load model")
    model = HybridCBMTrain(config=config, select_concept_data=select_concept_data)

    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    kltn_utils.rank_zero_info_newline("Test model")
    tester.test(model=model, ckpt_path=config.best_model, dataloaders=test_loader)

    print("Done")


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
