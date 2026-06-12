import torch
import torch.nn.functional as F
from kltn_utils import kltn_const, kltn_utils
from torch import nn

from ..adacbm_hybridcbm import adacbm_hybridcbm


class ExplicdHybridCBM(adacbm_hybridcbm.AdaHybridCBM):
    def __init__(self, config, select_concept_data):
        super().__init__(config, select_concept_data)

        clip_model_config = kltn_const.CLIP_MODELS[config.model.clip_model]
        visual_feature_dim = clip_model_config["visual_feature_dim"]
        num_heads = clip_model_config["num_heads"]

        ## Get visual_tokens
        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.num_concept, visual_feature_dim))
        )

        ## Get adaptive module
        self.adaptive_module = adacbm_hybridcbm.AdaptiveModule(
            dim=visual_feature_dim, num_layers=config.model.num_ada_layer
        )

        ## Get cross_attn
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = FFN(visual_feature_dim, visual_feature_dim * 4)
        self.layer_norm = nn.LayerNorm(visual_feature_dim)

        concept_dim = self.static_concept_feat.shape[1]
        self.proj = nn.Linear(
            in_features=visual_feature_dim, out_features=concept_dim, bias=False
        )

        # hook
        self.visual_features = None
        visual_feature_layer = get_visual_feature_layer(
            self.clip_model, config.model.clip_model
        )
        visual_feature_layer.register_forward_hook(self.hook_fn)

    def setup_grad(self):
        unfreeze_visual_encoder(self.clip_model, self.config.model.clip_model)

    def hook_fn(self, module, input, output):
        """
        Forward hook to capture patch-level visual features
        output shape: (B, 1 + N_patches, D)
        """
        self.visual_features = output

    def forward(self, img):
        batch_size = img.shape[0]

        # Get concept feat
        concept_feat = torch.cat(
            [self.static_concept_feat, self.dynamic_concept_feat],
            dim=0,
        )
        concept_feat = F.normalize(concept_feat, dim=1)

        ## Get agg_visual_tokens
        self.clip_model(img, None)
        img_feat_map = self.visual_features[:, 1:, :]
        img_feat_map = self.adaptive_module(img_feat_map)

        visual_tokens = self.visual_tokens.repeat(batch_size, 1, 1)
        agg_visual_tokens, _ = self.cross_attn(
            visual_tokens, img_feat_map, img_feat_map
        )
        agg_visual_tokens = self.layer_norm(self.ffn(agg_visual_tokens))
        agg_visual_tokens = self.proj(agg_visual_tokens.mean(dim=1))
        agg_visual_tokens = F.normalize(agg_visual_tokens, dim=1)

        ## Get hybrid_concept_logits
        hybrid_concept_logits = self.scale * (agg_visual_tokens @ concept_feat.T)

        ## Get label_logits
        label_logits = self.cls_head(hybrid_concept_logits)

        # Get concept_logits
        concept_logits = hybrid_concept_logits[:, : self.num_static_concept]

        return label_logits, concept_logits


class FFN(nn.Module):
    def __init__(self, input_dim, ff_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))

        return x


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
        kltn_utils.unfreeze_module(model.visual.trunk)
        kltn_utils.freeze_module(model.text)

    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        kltn_utils.unfreeze_module(model.visual)
        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)
