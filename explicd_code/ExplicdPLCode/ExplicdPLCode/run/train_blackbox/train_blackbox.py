from optparse import OptionParser

from kltn_utils import kltn_utils, train
from pytorch_lightning.utilities import rank_zero_info

from ...model.blackbox.train import BlackboxTrain, CustomMetric
from ..train_explicd import utils as train_explicd_utils
from . import utils as train_blackbox_utils


def main(config):
    train_explicd_utils.setup_train(config)

    rank_zero_info("Load model")
    model = BlackboxTrain(
        CustomMetric=CustomMetric, cp_path=config.cp_path, config=config
    )

    rank_zero_info("Load train, val dataset")
    train_transform, val_transform = kltn_utils.build_transform(config.transform_method)
    train_loader = train_blackbox_utils.load_dataset(config, train_transform, "train")
    val_loader = train_blackbox_utils.load_dataset(config, val_transform, "val")

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

    rank_zero_info("Done")


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
