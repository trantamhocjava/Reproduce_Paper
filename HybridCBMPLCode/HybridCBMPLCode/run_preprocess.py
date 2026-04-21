from optparse import OptionParser

from kltn_utils import kltn_utils

from .preprocess.coco import coco
from .preprocess.concept_net import concept_net


def main(config):
    config.device = "cuda"

    kltn_utils.rank_zero_info_newline("run coco")
    coco(config)

    kltn_utils.rank_zero_info_newline("run concept_net")
    concept_net(config)


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--clip_model",
        type="str",
        dest="clip_model",
    )
    parser.add_option(
        "--coco_dataset_dir",
        type="str",
        dest="coco_dataset_dir",
    )
    parser.add_option(
        "--concept_net_dataset_dir",
        type="str",
        dest="concept_net_dataset_dir",
    )
    parser.add_option(
        "--batch_size",
        type="int",
        dest="batch_size",
    )
    parser.add_option(
        "--context_length",
        type="int",
        dest="context_length",
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
