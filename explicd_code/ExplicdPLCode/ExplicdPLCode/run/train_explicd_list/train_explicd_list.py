from optparse import OptionParser

from kltn_utils import kltn_utils, train

from ...model.explicd_list.train import ExplicdListTrain, MetricCalculator
from ..train_explicd import utils as train_explicd_utils


def main(config):
    train_explicd_utils.setup_train(config)

    kltn_utils.rank_zero_info_newline("LOAD MODEL")
    model = ExplicdListTrain(
        CustomMetric=MetricCalculator, cp_path=config.cp_path, config=config
    )

    kltn_utils.rank_zero_info_newline("LOAD DATASET")
    train_transform, val_transform = kltn_utils.build_transform(config.transform)
    train_loader = train_explicd_utils.load_dataset(config, train_transform, "train")
    val_loader = train_explicd_utils.load_dataset(config, val_transform, "val")

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

    kltn_utils.rank_zero_info_newline("DONE")


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
