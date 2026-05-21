import os

from kltn_utils import kltn_const, kltn_utils

from . import const


def prepare_for_run_code(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(kltn_const.CP_PATH, exist_ok=True)

    config.class_concept = const.CLASS_AND_CONCEPT[config.dataset_name]
    config.concepts = config.class_concept["concepts"]
    config.class_names = config.class_concept["class_names"]
    config.num_class = len(config.class_names)
    config.concept2class = config.class_concept["concept2class"]

    config.num_images_per_class = kltn_utils.get_num_images_per_class(
        dataset_dir=config.dataset_dir, class_names=config.class_names, mode="train"
    )
