import os
import shutil
from optparse import OptionParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .train import BlackBoxTrain


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    utils.seed_everything_in_pl()

    os.makedirs(config.cp_path, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    rank_zero_info("Load model")
    model = BlackBoxTrain(config=config)

    rank_zero_info("Load train, val dataset")
    class2concept = const.CLASS2CONCEPT[config.dataset_name]

    trainLoader, valLoader, _ = utils.load_train_val_test(config, class2concept)
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

    rank_zero_info("Train Blackbox")
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

    utils.destroy_process_group()

    rank_zero_info("Done")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--last_state",
        type="str",
        dest="last_state",
        default=None,
    )
    parser.add_option(
        "--model",
        type="str",
        dest="model",
        help="[vit_base_patch16_224.orig_in21k, resnet50.a1_in1k]",
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
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
    )
    parser.add_option("--optimizer", dest="optimizer", default="adamw", type="str")
    parser.add_option(
        "--lr", dest="lr", default=0.0001, type="float", help="learning rate"
    )
    parser.add_option(
        "--dataset_name",
        type="str",
        dest="dataset_name",
    )
    parser.add_option(
        "--dataset_dir", type="str", dest="dataset_dir", help="the path of the dataset"
    )
    parser.add_option(
        "--amp", action="store_true", dest="amp", help="if use mixed precision training"
    )
    parser.add_option(
        "--scheduler",
        type="str",
        dest="scheduler",
        default=None,
        help="[LinearLR, ReduceLROnPlateau, StepLR]",
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
