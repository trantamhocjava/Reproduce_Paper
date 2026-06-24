import torch
import torch.nn.functional as F
from kltn_utils import kltn_const, kltn_utils
from torch import nn

from ..adacbm_hybridcbm import adacbm_hybridcbm
from ..explicd_hybridcbm.ffn import FFN
from . import utils as explicd_hybridcbm_utils


class ExplicdHybridCBM_v1(nn.Module):
    def __init__(self, config, select_concept_data):
        super().__init__()
        self.config = config

        # var
        self.register_buffer(
            "scale", torch.tensor(config.model.scale, dtype=torch.float32)
        )
        self.register_buffer(
            "static_concept_feat",
            select_concept_data["concept_feat"],
        )
        self.register_buffer(
            "class_feat",
            select_concept_data["class_feat"],
        )

        self.num_static_concept, concept_dim = select_concept_data["concept_feat"].shape
        self.num_concept = self.num_static_concept + config.model.num_dynamic_concept

        # clip_model
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.model.clip_model
        )

        # cls_head
        self.cls_head = torch.nn.Linear(self.num_concept, self.config.num_class)

        # dynamic_concept_feat
        self.dynamic_concept_feat = nn.Parameter(
            torch.randn(config.model.num_dynamic_concept, concept_dim)
        )

        # visual_tokens
        clip_model_config = kltn_const.CLIP_MODELS[config.model.clip_model]
        visual_feature_dim = clip_model_config["visual_feature_dim"]
        num_heads = clip_model_config["num_heads"]

        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(self.num_concept, visual_feature_dim))
        )

        ## adaptive module
        self.adaptive_module = adacbm_hybridcbm.AdaptiveModule(
            dim=visual_feature_dim, num_layers=config.model.num_ada_layer
        )

        ## cross_attn
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = FFN(visual_feature_dim, visual_feature_dim * 4)
        self.layer_norm = nn.LayerNorm(visual_feature_dim)

        self.proj = nn.Linear(
            in_features=visual_feature_dim, out_features=concept_dim, bias=False
        )

        # hook
        self.visual_features = None
        visual_feature_layer = explicd_hybridcbm_utils.get_visual_feature_layer(
            self.clip_model, config.model.clip_model
        )
        visual_feature_layer.register_forward_hook(self.hook_fn)

    def setup_grad(self):
        explicd_hybridcbm_utils.unfreeze_visual_encoder(
            self.clip_model, self.config.model.clip_model
        )

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
        explicd_hybridcbm_utils.forward_clip_model(
            img, self.clip_model, self.config.model.clip_model
        )

        img_feat_map = self.visual_features[:, 1:, :]
        img_feat_map = explicd_hybridcbm_utils.process_img_feat_map(
            img_feat_map, self.config.model.clip_model
        )

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
