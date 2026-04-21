from optparse import OptionParser

from kltn_utils.kltn_utils import print_demo_user_lib


def main(config):
    print_demo_user_lib(config.text)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "--text",
        type="str",
        dest="text",
        default=None,
    )
    (cfg, args) = parser.parse_args()

    main(cfg)
