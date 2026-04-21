import os
import shutil
from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from . import const, utils
from .train import ExplicdTrain


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    kltn_utils.rank_zero_info_newline("LOAD MODEL")
    model = ExplicdTrain(config=config)

    kltn_utils.rank_zero_info_newline("LOAD DATASET")
    class2concept = torch.tensor(
        const.CLASS2CONCEPT[config.dataset_name], dtype=torch.long
    )

    trainLoader, valLoader, _ = utils.load_train_val_test(config, class2concept)
    kltn_utils.rank_zero_info_newline("TRAIN")
    utils.print_shape_first_batch(trainLoader)
    kltn_utils.rank_zero_info_newline("VALIDATION")
    utils.print_shape_first_batch(valLoader)

    if config.last_state is not None:
        kltn_utils.rank_zero_info_newline(
            f"RESTORE LAST STATE FROM {config.last_state}"
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
    csv_logger = CSVLogger(save_dir=const.CP_PATH, name="", version=const.CSV_LOGS)

    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        max_epochs=config.end_epoch,
        precision="16-mixed" if config.amp else 32,
        strategy=DDPStrategy(find_unused_parameters=True),
        default_root_dir=const.CP_PATH,
        num_sanity_val_steps=0,
        logger=[csv_logger],
        callbacks=[model_ckpt],
    )

    kltn_utils.rank_zero_info_newline("TRAIN EXPLICD")
    trainer.fit(
        model,
        train_dataloaders=trainLoader,
        val_dataloaders=valLoader,
        ckpt_path=ckpt_path,
    )

    kltn_utils.rank_zero_info_newline("RESULT OF BEST MODEL ON VALSET")
    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    tester.test(
        model=model, ckpt_path=f"{const.CP_PATH}/best.ckpt", dataloaders=valLoader
    )

    kltn_utils.rank_zero_info_newline("DONE")


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
        "--transform", dest="transform", type="str", help="[paper, follow_backbone]"
    )
    parser.add_option("--optimizer", dest="optimizer", default="adamw", type="str")
    parser.add_option(
        "--lr", dest="lr", default=0.0001, type="float", help="learning rate"
    )
    parser.add_option("--weight_decay", dest="weight_decay", type="float")
    parser.add_option(
        "--scheduler",
        type="str",
        dest="scheduler",
        default=None,
        help="[LinearLR, ReduceLROnPlateau, StepLR]",
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
        "--clip_model",
        type="str",
        dest="clip_model",
        help="[hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K]",
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
