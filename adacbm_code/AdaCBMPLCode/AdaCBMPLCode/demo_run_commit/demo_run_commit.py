from optparse import OptionParser

import torch
from kltn_utils import kltn_utils


def demo_run_commit(config):
    x = torch.rand(config.num_rows, config.num_cols)
    y = torch.rand(config.num_rows + 5, config.num_cols + 5)

    torch.save(
        {
            "x": x,
            "y": y,
        },
        config.cp_path,
    )

    print(f"save {config.cp_path} ok")


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    config, args = parser.parse_args()
    config = kltn_utils.read_json_to_namespace(config.arg_json)
    demo_run_commit(config)

    kltn_utils.rank_zero_info_newline("DONE")
