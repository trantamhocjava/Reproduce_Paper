import os
import shutil
from optparse import OptionParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_info

from . import const, utils
from .train import SCBMTrainPL


def main(config):
    # os.environ["CUDA_VISIBLE_DEVICES"] = const.GPU
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    utils.seed_everything_in_pl()

    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]

    # utils.print_dist_0("Load model")
    rank_zero_info("Load model")
    model = SCBMTrainPL(config=config)

    # utils.print_dist_0("Load train, val dataset")
    rank_zero_info("Load train, val dataset")
    img2attr = utils.load_img2attr(config)

    trainLoader, valLoader, _ = utils.load_train_val_test(
        config, model.model.preprocess_list, img2attr
    )
    rank_zero_info("Train")
    utils.print_shape_first_batch(trainLoader)
    rank_zero_info("Val")
    utils.print_shape_first_batch(valLoader)

    rank_zero_info("init covariance")
    model.init_covariance(trainLoader)

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
        strategy=config.train_strategy,
        default_root_dir=const.CP_PATH,
        num_sanity_val_steps=0,
        logger=[csv_logger],
        callbacks=[model_ckpt],
    )

    rank_zero_info("Train SCBM")
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
    parser.add_option(
        "--optimizer", dest="optimizer", type="str", help="[sgd, adam, adamw]"
    )
    parser.add_option("--lr", dest="lr", type="float")
    parser.add_option(
        "--dataset_name",
        type="str",
        dest="dataset_name",
    )
    parser.add_option("--dataset_dir", type="str", dest="dataset_dir")
    parser.add_option("--train_strategy", type="str", dest="train_strategy")
    parser.add_option("--amp", action="store_true", dest="amp")
    parser.add_option(
        "--use_scheduler",
        action="store_true",
        dest="use_scheduler",
    )
    parser.add_option(
        "--scheduler",
        type="str",
        dest="scheduler",
        default=None,
        help="[LinearLR, ReduceLROnPlateau]",
    )
    parser.add_option(
        "--num_concepts",
        type="int",
        dest="num_concepts",
    )
    parser.add_option(
        "--head_arch", type="str", dest="head_arch", help="[linear, nonlinear]"
    )
    parser.add_option("--alpha", type="float", dest="alpha", default=None)
    parser.add_option(
        "--encoder_arch",
        type="str",
        dest="encoder_arch",
        help="[resnet18, simple_CNN, FCNN]",
    )
    parser.add_option(
        "--decrease_every",
        type="int",
        dest="decrease_every",
    )
    parser.add_option(
        "--lr_divisor",
        type="int",
        dest="lr_divisor",
    )
    parser.add_option(
        "--weight_decay",
        type="float",
        dest="weight_decay",
    )
    parser.add_option(
        "--compile",
        action="store_true",
        dest="compile",
    )
    parser.add_option(
        "--num_monte_carlo",
        type="int",
        dest="num_monte_carlo",
    )
    parser.add_option(
        "--straight_through",
        action="store_true",
        dest="straight_through",
    )
    parser.add_option(
        "--concept_learning", type="str", dest="concept_learning", default=None
    )
    parser.add_option("--inter_policy", type="str", dest="inter_policy", default=None)
    parser.add_option(
        "--inter_strategy", type="str", dest="inter_strategy", default=None
    )
    parser.add_option(
        "--pretrain_concepts", action="store_true", dest="pretrain_concepts"
    )
    parser.add_option(
        "--embedding_size", type="int", dest="embedding_size", default=None
    )
    parser.add_option("--cov_type", type="str", dest="cov_type", default=None)
    parser.add_option("--reg_precision", type="str", dest="reg_precision", default=None)
    parser.add_option("--reg_weight", type="float", dest="reg_weight", default=None)
    parser.add_option("--level", type="float", dest="level", default=None)

    (cfg, args) = parser.parse_args()

    main(cfg)
