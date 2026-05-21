import os

import numpy as np
from kltn_utils import kltn_utils

from ... import const


class ConceptProcessor:
    def __init__(self, concepts, concept2class, class_names) -> None:
        self.concepts = np.array(concepts)
        self.concept2class = np.array(concept2class)
        self.class_names = np.array(class_names)

    def next(self):
        unique_concepts, unique_idx = np.unique(self.concepts, return_index=True)
        is_good = self.check_no_cls_names(unique_concepts, self.class_names)

        left_idx = unique_idx[is_good]

        concepts = self.concepts[left_idx].tolist()
        concept2class = self.concept2class[left_idx].tolist()

        return concepts, concept2class

    def has_pattern(self, concepts, pattern):
        """
        Return a boolean array where it is true if one concept contains the pattern
        """
        return np.char.find(concepts, pattern) != -1

    def check_no_cls_names(self, concepts, cls_names):
        res = np.ones(len(concepts), dtype=bool)

        for cls_name in cls_names:
            no_cls_name = ~self.has_pattern(concepts, cls_name)
            res = res & no_cls_name

        return res


def setup_train(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(config.cp_path, exist_ok=True)

    config.class_concept = const.CLASS_AND_CONCEPT[config.dataset_name]

    config.concepts = config.class_concept["concepts"]

    config.class_names = config.class_concept["class_names"]
    config.num_class = len(config.class_names)

    config.concept2class = config.class_concept["concept2class"]
    config.num_images_per_class = kltn_utils.get_num_images_per_class(
        dataset_dir=f"{config.dataset_dir}/train", class_names=config.class_names
    )
