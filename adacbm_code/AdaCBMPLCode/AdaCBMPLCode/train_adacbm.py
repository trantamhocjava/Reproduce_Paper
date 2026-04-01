import os
import shutil
from optparse import OptionParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .train import AdacbmTrain


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    rank_zero_info("Load model")
    select_concepts_data = torch.load(
        config.select_concepts_data_path, map_location="cpu"
    )

    model = AdacbmTrain(select_concepts_data=select_concepts_data, config=config)

    rank_zero_info("Load train, val dataset")
    trainLoader, valLoader, _ = utils.load_train_val_test(config)
    rank_zero_info("Train")
    utils.print_shape_first_batch(trainLoader)
    rank_zero_info("Val")
    utils.print_shape_first_batch(valLoader)

    if config.last_state is not None:
        rank_zero_info(f"Restore last state from {config.last_state}")
        ckpt_path = f"{config.last_state}/last.ckpt"
        shutil.copy(f"{config.last_state}/best.ckpt", f"{const.CP_PATH}/best.ckpt")
    else:
        ckpt_path = None

    model_ckpt = ModelCheckpoint(
        dirpath=const.CP_PATH,
        save_top_k=1,
        save_last=True,
        monitor=config.monitor,
        mode=utils.get_mode(config.monitor),
        filename="best",
    )
    csv_logger = CSVLogger(save_dir=const.CP_PATH, name="", version=const.CSV_LOGS)

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

    rank_zero_info("Train Adacbm")
    trainer.fit(
        model,
        train_dataloaders=trainLoader,
        val_dataloaders=valLoader,
        ckpt_path=ckpt_path,
    )

    rank_zero_info("Result of best model on valset")
    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    tester.test(
        model=model, ckpt_path=f"{const.CP_PATH}/best.ckpt", dataloaders=valLoader
    )

    rank_zero_info("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--select_concepts_data_path",
        dest="select_concepts_data_path",
        type="str",
    )
    parser.add_option(
        "--last_state",
        type="str",
        dest="last_state",
        default=None,
    )
    parser.add_option(
        "--monitor",
        type="str",
        dest="monitor",
    )
    parser.add_option(
        "--start_epoch",
        dest="start_epoch",
        type="int",
    )
    parser.add_option(
        "--end_epoch",
        dest="end_epoch",
        type="int",
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
    )
    parser.add_option(
        "--dataset_name",
        dest="dataset_name",
        type="str",
    )
    parser.add_option(
        "--dataset_dir", type="str", dest="dataset_dir", help="the path of the dataset"
    )
    parser.add_option(
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
    )
    parser.add_option("--optimizer", dest="optimizer", default="adamw", type="str")
    parser.add_option(
        "--lr", dest="lr", default=0.0001, type="float", help="learning rate"
    )
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )
    parser.add_option(
        "--amp",
        action="store_true",
        dest="amp",
    )
    parser.add_option("--clip_model", dest="clip_model", type="str")
    parser.add_option("--num_layers", dest="num_layers", type="int")

    (cfg, args) = parser.parse_args()

    main(cfg)
