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
        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x


def get_prefix(config, key):
    if config.dataset_name == "isic2018":
        prefix = f"the {key} of the lesion is "
    elif config.dataset_name == "IDRID":
        prefix = f"the {key} of the retina is "
    elif config.dataset_name == "BUSI":
        prefix = f"the {key} of the breast is "
    elif config.dataset_name == "nct_crc_he":
        prefix = f"the {key} of the tissue is "
    elif config.dataset_name == "lcc":
        prefix = f"the {key} of the tissue is "

    return prefix


def get_visual_feature_layer(model, model_name):
    if model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        visual_feature_layer = model.visual.trunk.blocks[-1]
    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        visual_feature_layer = model.visual.transformer.resblocks[-1]

    return visual_feature_layer


def freeze_grad_for_clip_model(model, model_name):
    if model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        kltn_utils.freeze_module(model.text)

    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)


def get_concept_feat_dict(clip_model_name, concept_dict, config):
    clip_model, tokenizer = kltn_utils.build_clip_model(clip_model_name)

    concept_feat_dict = {}
    for key in concept_dict.keys():
        prefix = get_prefix(config, key)
        prefix_concept_list = [prefix + concept for concept in concept_dict[key]]

        concept_feat = kltn_utils.get_txt_feat(
            prefix_concept_list,
            clip_model,
            clip_model_name,
            tokenizer,
            config.batch_size,
        )
        concept_feat_dict[key] = F.normalize(concept_feat, dim=-1)

    return concept_feat_dict
