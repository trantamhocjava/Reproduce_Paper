from optparse import OptionParser

from kltn_utils import kltn_utils
from pytorch_lightning import Trainer

from ... import utils
from ...models.translator.train import ConceptTranslatorTrain
from . import utils as train_translator_utils


def main(config):
    utils.prepare_for_run_code(config)

    kltn_utils.rank_zero_info_newline("Load test dataset")
    test_loader = train_translator_utils.load_dataset(config, "test")

    kltn_utils.rank_zero_info_newline("Load model")
    model = ConceptTranslatorTrain(config=config)

    kltn_utils.rank_zero_info_newline("Test model")
    tester = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    tester.test(model=model, ckpt_path=config.best_model, dataloaders=test_loader)

    print("Done")


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
