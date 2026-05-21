import torch
import torch.nn.functional as F
from kltn_utils import kltn_const, kltn_utils
from torch import nn

from ..adacbm_hybridcbm import adacbm_hybridcbm


class ExplicdHybridCBM(nn.Module):
    def __init__(self, config, concept_feat, concept2class):
        super().__init__()
        self.config = config

        num_concept, concept_dim = concept_feat.shape

        # module
        clip_model_config = kltn_const.CLIP_MODELS[config.hybridcbm.clip_model]
        visual_feature_dim = clip_model_config["visual_feature_dim"]
        num_heads = clip_model_config["num_heads"]

        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.hybridcbm.clip_model
        )
        self.cls_head = adacbm_hybridcbm.get_cls_head(
            cls_head_type=config.hybridcbm.cls_head_type,
            num_concept=num_concept,
            num_class=config.num_class,
            concept2class=concept2class,
        )
        self.adaptive_layer = AdaptiveModule(
            dim=visual_feature_dim,
            num_layers=config.hybridcbm.ada_num_layer,
        )

        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(config.num_concept, visual_feature_dim))
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = FFN(visual_feature_dim, visual_feature_dim * 4)
        self.layer_norm = nn.LayerNorm(visual_feature_dim)
        self.proj = nn.Linear(
            in_features=visual_feature_dim, out_features=concept_dim, bias=False
        )

        # var
        self.register_buffer(
            "scale",
            torch.tensor(
                kltn_const.CLIP_MODELS[config.hybridcbm.clip_model]["logit_scale"]
            ),
        )
        self.register_buffer(
            "concept_feat",
            concept_feat,
        )

        # hook
        self.visual_features = None
        visual_feature_layer = get_visual_feature_layer(
            self.clip_model, config.hybridcbm.clip_model
        )
        visual_feature_layer.register_forward_hook(self.hook_fn)

        # grad
        unfreeze_visual_encoder(self.clip_model, config.hybridcbm.clip_model)

    def hook_fn(self, module, input, output):
        """
        Forward hook to capture patch-level visual features
        output shape: (B, 1 + N_patches, D)
        """
        self.visual_features = output

    def forward(self, img):
        batch_size = img.shape[0]

        self.clip_model(img, None)
        img_feat_map = self.visual_features[:, 1:, :]
        img_feat_map = self.adaptive_layer(img_feat_map)

        visual_tokens = self.visual_tokens.repeat(batch_size, 1, 1)
        agg_visual_tokens, _ = self.cross_attn(
            visual_tokens, img_feat_map, img_feat_map
        )
        agg_visual_tokens = agg_visual_tokens.mean(dim=1)
        agg_visual_tokens = F.normalize(
            self.proj(self.layer_norm(self.ffn(agg_visual_tokens))), dim=-1
        )

        concept_logits = self.scale * (agg_visual_tokens @ self.concept_feat.T)
        label_logits = self.cls_head(concept_logits)

        return label_logits, concept_logits


class AdaptiveModule(nn.Module):
    def __init__(self, dim, num_layers=1):
        super().__init__()

        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.LeakyReLU())

        self.linear1 = nn.Sequential(*layers)

    def forward(self, org_img_feat):
        img_feat = self.linear1(org_img_feat)
        img_feat = img_feat + org_img_feat  # residual handling
        img_feat = F.normalize(img_feat, dim=-1)  # normalize

        return img_feat


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
        kltn_utils.freeze_module(model.text)

    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)
