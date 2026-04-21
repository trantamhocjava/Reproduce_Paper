import glob
import os

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.trainer import Trainer

from datasets.dataloader import DataBank
from models.cbm.linearCBM import LinearCBM
from models.clip import ClipEncoder
from models.conceptBank.hybrid_bank import HybridConceptBank
from utils.config import get_args

SEED = 42


def get_datamodule_fromconfig(config, clip_encoder=None):
    if clip_encoder is None:
        clip_encoder = ClipEncoder(model_name=config.clip_model)
        clip_encoder.eval()
        clip_encoder.to(config.device)

    return DataBank(
        data_root=config.data_root,
        exp_root=config.exp_root,
        # concept
        n_shots=config.n_shots,
        # clip
        use_img_features=config.use_img_features,
        clip_encoder=clip_encoder,
        # dataloader
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        force_compute=config.force_compute,
    )


def get_concept_bank_fromconfig(config):
    return HybridConceptBank(
        exp_root=config.exp_root,
        data_root=config.data_root,
        # concept
        num_static_concept=config.num_static_concept,
        num_dynamic_concept=config.num_dynamic_concept,
        concept_select_fn=config.concept_select_fn,
        submodular_weights=config.submodular_weights,
        # clip
        clip_model=config.clip_model,
        translator_path=config.translator_path,
    )


def init_data_bank(config, captions):
    concept_bank = get_concept_bank_fromconfig(config)
    concept_bank.to(torch.device(config.device))
    datamodule = get_datamodule_fromconfig(
        config, clip_encoder=concept_bank.clip_encoder
    )
    concept_bank.initialize(
        img_features=datamodule.img_features["train"],
        num_images_per_class=datamodule.num_images_per_class,
        captions=captions,
    )
    concept_bank.to(torch.device(config.device))
    return concept_bank, datamodule


def load_checkpoint(config):
    checkpoint_dir = config.exp_root.joinpath("checkpoints")
    if config.use_last_ckpt:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return None
        checkpoint_path = max(checkpoints, key=os.path.getctime)
    else:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*val_acc*.ckpt"))
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return None

        def get_val_acc(ckpt):
            filename = os.path.basename(ckpt)
            val_acc = filename.split("val_acc=")[-1].split(".ckpt")[0].split("-")[0]
            return float(val_acc)

        checkpoint_path = max(checkpoints, key=get_val_acc)
        return checkpoint_path


def main():
    config = get_args()
    seed_everything(SEED)

    if config.test:
        concept_bank, datamodule = init_data_bank(config, captions=None)

        checkpoint_path = load_checkpoint(config)
        if checkpoint_path is None:
            print("No checkpoint found. Exiting.")
            return

        model = LinearCBM.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(config.device),
            config=config,
            conceptbank=concept_bank,
            strict=False,
        )
        model.to(config.device)
        model.eval()
        trainer = Trainer(
            devices=[0],
            callbacks=[TQDMProgressBar(refresh_rate=1)],
            default_root_dir=config.exp_root,
        )
        print("Testing the model...")
        trainer.test(model, datamodule=datamodule)

    else:
        project_name = config.dataset
        concept_bank, datamodule = init_data_bank(config, captions=None)
        model = LinearCBM(config, concept_bank)

        check_interval = 10
        checkpoint_dir = config.exp_root.joinpath("checkpoints")

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{step}-{val_acc:.4f}",
            monitor="val_acc",
            mode="max",
            save_top_k=3,
            every_n_epochs=check_interval,
        )

        trainer = Trainer(
            accelerator="auto",
            devices=[0],
            callbacks=[
                checkpoint_callback,
                TQDMProgressBar(refresh_rate=1),
                LearningRateMonitor(),
            ],
            check_val_every_n_epoch=check_interval,
            default_root_dir=config.exp_root,
            max_epochs=config.max_epochs,
            log_every_n_steps=10,
        )
        print("Training the model...")
        trainer.fit(model, datamodule=datamodule, ckpt_path=load_checkpoint(config))


if __name__ == "__main__":
    main()
