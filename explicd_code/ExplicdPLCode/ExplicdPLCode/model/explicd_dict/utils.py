import torch.nn.functional as F
from kltn_utils import kltn_utils
from torch import nn


class FFN(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))

        return x


def get_final_concept(config, key, concept):
    if config.dataset_name == "isic2018":
        concept = f"the {key} of the lesion is {concept}"
    elif config.dataset_name == "IDRID":
        concept = f"the {key} of the retina is {concept}"
    elif config.dataset_name == "BUSI":
        concept = f"the {key} of the breast is {concept}"
    elif config.dataset_name == "nct_crc_he":
        concept = f"the {key} of the tissue is {concept}"
    elif config.dataset_name == "lcc":
        concept = f"the {key} of the tissue is {concept}"

    return concept


def get_visual_feature_layer(model, model_name):
    if model_name in (
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k",
    ):
        visual_feature_layer = model.visual.trunk.blocks[-1]
    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        visual_feature_layer = model.visual.transformer.resblocks[-1]

    return visual_feature_layer


def unfreeze_visual_encoder(model, model_name):
    if model_name == (
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k",
    ):
        kltn_utils.freeze_module(model.text)

    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)


def get_concept_feat_dict(clip_model_name, concept_dict, config):
    clip_model, tokenizer = kltn_utils.build_clip_model(clip_model_name)

    concept_feat_dict = {}
    for key in concept_dict.keys():
        concept_list = [
            get_final_concept(config, key, concept) for concept in concept_dict[key]
        ]

        concept_feat = kltn_utils.get_txt_feat(
            concept_list,
            clip_model,
            clip_model_name,
            tokenizer,
            config.batch_size,
        )
        concept_feat_dict[key] = F.normalize(concept_feat, dim=-1)

    return concept_feat_dict
