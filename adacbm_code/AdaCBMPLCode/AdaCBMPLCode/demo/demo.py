from optparse import OptionParser

from .. import const
from ..const import CLASS_NAMES


def main(config):
    print(f"CLASS_NAMES: {CLASS_NAMES}")
    print(f"DATASET_CLASS: {const.DATASET_CLASS}")
    print(f"config.n_shots: {config.n_shots}")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--n_shots",
        dest="n_shots",
        default=-1,
        type="int",
    )

    (cfg, args) = parser.parse_args()
    main(cfg)
