from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from kltn_utils.select_concept import select_concept

from . import utils as run_select_concept_utils
from .utils import ConceptProcessor


def main(config):
    run_select_concept_utils.setup_train(config)

    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    train_transform, _ = kltn_utils.build_transform(config.transform)
    img_feat, _ = kltn_utils.get_img_feat(
        clip_model,
        config.clip_model,
        f"{config.dataset_dir}/train",
        config.batch_size,
        train_transform,
        config.class_names,
    )

    concepts, concept2class = ConceptProcessor(
        concepts=config.concepts,
        concept2class=config.concept2class,
        class_names=config.class_names,
    ).next()
    concept_feat = kltn_utils.get_txt_feat(
        concepts, clip_model, config.clip_model, tokenizer, config.batch_size
    )

    class_feat = kltn_utils.get_txt_feat(
        config.class_names, clip_model, config.clip_model, tokenizer, config.batch_size
    )

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

    torch.save(
        {
            "concept_feat": concept_feat,
            "concept2class": concept2class,
            "concepts": concepts,
            "class_feat": class_feat,
        },
        f"{config.cp_path}/select_concept_data.pth",
    )


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    config, args = parser.parse_args()
    config = kltn_utils.read_json_to_namespace(config.arg_json)
    main(config)

    kltn_utils.rank_zero_info_newline("DONE")
