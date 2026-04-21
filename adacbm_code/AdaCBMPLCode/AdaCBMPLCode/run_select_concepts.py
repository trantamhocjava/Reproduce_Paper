import os
from optparse import OptionParser

import torch
from kltn_utils import kltn_const, kltn_utils
from pytorch_lightning.utilities import rank_zero_info
from torchvision.transforms import v2

from . import const
from .select_concept import utils as select_concept_utils
from .select_concept.select_algo import paper_selection


def main(config):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    kltn_utils.seed_everything_in_pl()
    os.makedirs(const.CP_PATH, exist_ok=True)

    config.class_names = const.CLASS_NAMES[config.dataset_name]
    config.class2concepts = const.CLASS2CONCEPT[config.dataset_name]

    kltn_utils.rank_zero_info_newline("build clip model")
    clip_model, clip_tokenizer = kltn_utils.build_clip_model(config.clip_model)

    transform = v2.Compose(kltn_const.PREPROCESS_LIST)

    kltn_utils.rank_zero_info_newline("build image feat")
    num_images_per_class = select_concept_utils.get_num_images_per_class(config)
    img_feat, label = select_concept_utils.prepare_img_feat(
        clip_model, transform, config
    )
    kltn_utils.rank_zero_info_newline(f"img_feat.shape: {img_feat.shape}")
    kltn_utils.rank_zero_info_newline(f"label.shape: {label.shape}")

    kltn_utils.rank_zero_info_newline("build concept feat")
    all_concepts, concept2cls = select_concept_utils.get_all_concepts_and_concept2cls(
        config
    )
    kltn_utils.rank_zero_info_newline(f"all_concepts.shape: {all_concepts.shape}")

    all_concepts, concept2cls = select_concept_utils.preprocess_concept(
        all_concepts, config, concept2cls
    )
    kltn_utils.rank_zero_info_newline(
        f"after preprocess_concept: all_concepts.shape: {all_concepts.shape}"
    )
    concept_feat = select_concept_utils.prepare_concept_feat(
        clip_model, all_concepts, config, clip_tokenizer
    )
    kltn_utils.rank_zero_info_newline(f"concept_feat.shape: {concept_feat.shape}")

    kltn_utils.rank_zero_info_newline("get selected concept idx")
    select_idx = paper_selection(
        img_feat.cpu(),
        concept_feat.cpu(),
        concept2cls,
        config.num_selected_concept,
        num_images_per_class,
        pearson_weight=config.pearson_weight,
    )
    select_concept_feat = concept_feat[select_idx]
    select_all_concepts = all_concepts[select_idx]
    select_concept2cls = concept2cls[select_idx]
    kltn_utils.rank_zero_info_newline(f"select_idx.shape: {select_idx.shape}")

    save_data = {
        "select_idx": select_idx,
        "select_concept_feat": select_concept_feat,
        "select_all_concepts": select_all_concepts,
        "select_concept2cls": select_concept2cls,
    }
    torch.save(save_data, f"{const.CP_PATH}/save_data.pth")

    select_idx = [int(item) for item in list(select_idx)]
    kltn_utils.save_dict_to_json(
        {"select_idx": select_idx}, f"{const.CP_PATH}/select_idx.json"
    )

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
        help="""
        "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", 
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        """,
    )
    parser.add_option(
        "--batch_size",
        dest="batch_size",
        type="int",
    )
    parser.add_option(
        "--pearson_weight",
        dest="pearson_weight",
        type="float",
    )
    parser.add_option(
        "--num_selected_concept",
        dest="num_selected_concept",
        type="int",
    )

    (cfg, args) = parser.parse_args()

    main(cfg)
