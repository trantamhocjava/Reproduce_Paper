from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from kltn_utils.select_concept import select_concept

from . import utils as run_select_concept_utils
from .utils import ConceptProcessor


def main(config):
    run_select_concept_utils.setup_train(config)

    # TODO: DEBUG
    kltn_utils.rank_zero_info_newline("config.concept2class")
    kltn_utils.rank_zero_info_newline(config.concept2class)
    kltn_utils.rank_zero_info_newline(f"len(config.class_names: {config.class_names}")
    # END DEBUG

    kltn_utils.rank_zero_info_newline("Load CLIP model")
    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    kltn_utils.rank_zero_info_newline("Load img feat in train dataset")
    train_transform, _ = kltn_utils.build_transform(config.transform)
    img_feat, _ = kltn_utils.get_img_feat(
        clip_model,
        config.clip_model,
        f"{config.dataset_dir}/train",
        config.batch_size,
        train_transform,
        config.class_names,
    )

    # TODO: DEBUG
    kltn_utils.rank_zero_info_newline(f"img_feat.shape: {img_feat.shape}")
    # END DEBUG

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
    select_idx = select_concept.get_select_concept_idx(
        select_concept_method=config.select_concept_method,
        img_feat=img_feat,
        concept_feat=concept_feat,
        concept2class=concept2class,
        num_select_concepts=config.num_select_concepts,
        num_images_per_class=config.num_images_per_class,
        weight=config.weight,
    )
    concept_feat = concept_feat[select_idx]
    concept2class = kltn_utils.get_sublist(concept2class, select_idx)
    concepts = kltn_utils.get_sublist(concepts, select_idx)

    # TODO: DEBUG
    kltn_utils.rank_zero_info_newline(f"concept_feat.shape: {concept_feat.shape}")
    kltn_utils.rank_zero_info_newline("concept2class")
    kltn_utils.rank_zero_info_newline(concept2class)
    kltn_utils.rank_zero_info_newline("concepts")
    kltn_utils.rank_zero_info_newline(concepts)

    # END DEBUG

    kltn_utils.rank_zero_info_newline("Save selected concepts")
    torch.save(
        {
            "concept_feat": concept_feat,
            "concept2class": concept2class,
            "concepts": concepts,
            "class_feat": class_feat,
        },
        f"{config.cp_path}/select_concept_data.pth",
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
