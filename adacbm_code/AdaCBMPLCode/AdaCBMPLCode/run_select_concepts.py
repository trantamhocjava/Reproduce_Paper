import os
from optparse import OptionParser

import torch
from pytorch_lightning.utilities import rank_zero_info
from torchvision.transforms import v2

from . import const, utils
from .select_concept import utils as select_concept_utils
from .select_concept.select_algo import paper_selection


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]
    config.class2concepts = const.CLASS2CONCEPT[config.dataset_name]

    clip_model, clip_tokenizer = utils.build_clip_model(config.clip_model)
    transform = v2.Compose(const.PREPROCESS_LIST)

    num_images_per_class = select_concept_utils.get_num_images_per_class(config)

    img_feat, label = select_concept_utils.prepare_img_feat(
        clip_model, transform, config
    )

    all_concepts, concept2cls, num_concept = (
        select_concept_utils.get_all_concepts_and_concept2cls(config)
    )
    all_concepts, concept2cls = select_concept_utils.preprocess_concept(
        all_concepts, config, concept2cls
    )

    concept_feat = select_concept_utils.prepare_txt_feat(
        clip_model, all_concepts, config, clip_tokenizer
    )

    select_idx = paper_selection(
        img_feat,
        concept_feat,
        concept2cls,
        num_concept,
        num_images_per_class,
        pearson_weight=config.pearson_weight,
    )
    select_concept_feat = concept_feat[select_idx]
    select_all_concepts = all_concepts[select_idx]
    select_concept2cls = concept2cls[select_idx]

    save_data = {
        "select_idx": select_idx,
        "select_concept_feat": select_concept_feat,
        "select_all_concepts": select_all_concepts,
        "select_concept2cls": select_concept2cls,
    }
    torch.save(save_data, config.save_data_path)

    rank_zero_info("Done")


if __name__ == "__main__":
    print("run select_concepts")

    parser = OptionParser()
    parser.add_option(
        "--dataset_name",
        dest="dataset_name",
        type="str",
    )
    parser.add_option(
        "--dataset_dir",
        dest="dataset_dir",
        type="str",
    )
    parser.add_option(
        "--clip_model",
        dest="clip_model",
        type="str",
    )
    parser.add_option(
        "--pearson_weight",
        dest="pearson_weight",
        type="float",
    )
    parser.add_option(
        "--save_data_path",
        dest="save_data_path",
        type="str",
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
