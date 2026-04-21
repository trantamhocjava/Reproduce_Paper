import os
import shutil
from optparse import OptionParser

from kltn_utils import kltn_utils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from . import const
from .translator import utils as train_utils
from .translator.train import ConceptTranslatorTrain


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config.epochs = config.end_epoch - config.start_epoch + 1

    kltn_utils.rank_zero_info_newline("Load train dataset")
    trainLoader = train_utils.load_train(config)

    kltn_utils.rank_zero_info_newline("Load model")
    model = ConceptTranslatorTrain(config=config, n_batchs=len(trainLoader))

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
        mode="max",
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

    kltn_utils.rank_zero_info_newline("Train model")
    trainer.fit(
        model,
        train_dataloaders=trainLoader,
        ckpt_path=ckpt_path,
    )

    kltn_utils.rank_zero_info_newline("Result of best model on trainset")
    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    tester.test(
        model=model, ckpt_path=f"{const.CP_PATH}/best.ckpt", dataloaders=trainLoader
    )

    kltn_utils.rank_zero_info_newline("Done")


if __name__ == "__main__":
    parser = OptionParser()
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
        "--conceptPath",
        dest="conceptPath",
        type="str",
    )
    parser.add_option(
        "--cocoPath",
        dest="cocoPath",
        type="str",
    )
    parser.add_option("--optimizer", dest="optimizer", default="adamw", type="str")
    parser.add_option(
        "--lr", dest="lr", default=0.0001, type="float", help="learning rate"
    )
    parser.add_option(
        "--weight_decay", dest="weight_decay", type="float", help="weight decay"
    )
    parser.add_option(
        "--amp",
        action="store_true",
        dest="amp",
    )
    parser.add_option("--scheduler", dest="scheduler", type="str", default=None)

    (cfg, args) = parser.parse_args()

    main(cfg)
