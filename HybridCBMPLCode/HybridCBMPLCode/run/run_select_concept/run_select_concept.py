import os
from optparse import OptionParser

import torch
from kltn_utils import kltn_utils

from ... import const
from . import utils as run_select_concept_utils
from .utils import ConceptProcessor


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config = run_select_concept_utils.update_config(config)

    kltn_utils.rank_zero_info_newline("Load CLIP model")
    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    kltn_utils.rank_zero_info_newline("Load img feat in train dataset")
    train_transform, _ = kltn_utils.build_transform(config.transform_method)
    img_feat = kltn_utils.get_img_feat(
        clip_model,
        config.clip_model,
        f"{config.dataset_dir}/train",
        config.batch_size,
        train_transform,
        config.class_names,
    )

    kltn_utils.rank_zero_info_newline("Load concept feat")
    concepts, concept2class = ConceptProcessor(
        concepts=config.concepts,
        concept2class=config.concept2class,
        class_names=config.class_names,
    ).next()
    concept_feat = kltn_utils.get_txt_feat(
        concepts, clip_model, config.clip_model, tokenizer, config.batch_size
    )

    kltn_utils.rank_zero_info_newline("Load class feat")
    class_feat = kltn_utils.get_txt_feat(
        config.class_names, clip_model, config.clip_model, tokenizer, config.batch_size
    )

    kltn_utils.rank_zero_info_newline("get selected concept idx")
    select_idx = run_select_concept_utils.get_select_concept_idx(
        img_feat, concept_feat, config
    )
    concept_feat = concept_feat[select_idx]
    concept2class = kltn_utils.get_sublist(concept2class, select_idx)
    concepts = kltn_utils.get_sublist(concepts, select_idx)

    kltn_utils.rank_zero_info_newline("Save selected concepts")
    torch.save(
        {
            "concept_feat": concept_feat,
            "concept2class": concept2class,
            "concepts": concepts,
            "class_feat": class_feat,
        },
        f"{const.CP_PATH}/select_concept_data.pth",
    )

    kltn_utils.rank_zero_info_newline("Done")


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
