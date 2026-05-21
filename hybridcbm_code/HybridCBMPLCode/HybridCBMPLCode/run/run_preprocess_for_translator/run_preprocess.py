from optparse import OptionParser

from kltn_utils import kltn_utils

from ... import utils
from ...preprocess.coco import coco
from ...preprocess.concept_net import concept_net


def main(config):
    utils.prepare_for_run_code(config)

    kltn_utils.rank_zero_info_newline("run coco")
    coco(config)

    kltn_utils.rank_zero_info_newline("run concept_net")
    concept_net(config)


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
