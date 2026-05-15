import os
import shutil
from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

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

    kltn_utils.rank_zero_info_newline("Load train, val dataset")
    train_transform, val_transform = kltn_utils.build_transform(config.transform_method)
    train_loader = train_hybridcbm_utils.load_dataloader(
        config, train_transform, select_concept_data["concept2class"], "train"
    )
    val_loader = train_hybridcbm_utils.load_dataloader(
        config, val_transform, select_concept_data["concept2class"], "val"
    )

    kltn_utils.rank_zero_info_newline("Load model")
    model = HybridCBMTrain(config=config, select_concept_data=select_concept_data)

    if config.last_state is not None:
        kltn_utils.rank_zero_info_newline(
            f"Restore last state from {config.last_state}"
        )
        ckpt_path = f"{config.last_state}/last.ckpt"
        shutil.copy(f"{config.last_state}/best.ckpt", f"{const.CP_PATH}/best.ckpt")
    else:
        ckpt_path = None

    model_ckpt = ModelCheckpoint(
        dirpath=const.CP_PATH,
        save_top_k=1,
        save_last=True,
        monitor=config.monitor,
        mode=kltn_utils.get_mode(config.monitor),
        filename="best",
    )
    csv_logger = CSVLogger(save_dir=const.CP_PATH, name="", version="")

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=config.end_epoch,
        precision="16-mixed" if config.amp else 32,
        strategy="ddp",
        default_root_dir=const.CP_PATH,
        num_sanity_val_steps=0,
        logger=[csv_logger],
        callbacks=[model_ckpt],
    )

    kltn_utils.rank_zero_info_newline("Train model")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    kltn_utils.rank_zero_info_newline("Evaluate on val")
    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    tester.test(
        model=model, ckpt_path=f"{const.CP_PATH}/best.ckpt", dataloaders=val_loader
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
