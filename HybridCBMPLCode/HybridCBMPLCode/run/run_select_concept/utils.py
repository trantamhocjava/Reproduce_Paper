import numpy as np
from kltn_utils import kltn_utils

from ... import const, utils
from ...select_concept.adacbm_select_concept.select_algo import adacbm_selection
from ...select_concept.select_concept_algo import random_select, submodular_select


class ConceptProcessor:
    def __init__(self, concepts, concept2class, class_names) -> None:
        self.concepts = concepts
        self.concept2class = concept2class
        self.class_names = class_names

    def next(self):
        unique_concepts, unique_idx = np.unique(self.concepts, return_index=True)
        is_good = self.check_no_cls_names(unique_concepts, self.class_names)

        left_idx = unique_idx[is_good]

        concepts = kltn_utils.get_sublist(self.concepts, left_idx)
        concept2class = kltn_utils.get_sublist(self.concept2class, left_idx)

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


def get_select_concept_idx(img_feat, concept_feat, config):
    if config.select_concept_method == "submodular":
        select_idx = submodular_select(
            img_feat,
            concept_feat,
            config.concept2class,
            config.num_select_concepts,
            config.num_images_per_class,
            config.submodular_weights,
        )
    elif config.select_concept_method == "random":
        select_idx = random_select(
            config.concept2class,
            config.num_select_concepts,
            config.num_images_per_class,
        )
    elif config.select_concept_method == "adacbm":
        select_idx = adacbm_selection(
            img_feat,
            concept_feat,
            config.concept2class,
            config.num_select_concepts,
            config.num_images_per_class,
            config.pearson_weight,
        )

    return select_idx


def update_config(config):
    config.num_images_per_class = utils.get_num_images_per_class(config)
    config.class_concept = const.CLASS_AND_CONCEPT[config.dataset_name]
    config.concepts = config.class_concept["concepts"]
    config.class_names = config.class_concept["class_names"]
    config.num_class = len(config.class_names)
    config.concept2class = config.class_concept["concept2class"]

    return config
