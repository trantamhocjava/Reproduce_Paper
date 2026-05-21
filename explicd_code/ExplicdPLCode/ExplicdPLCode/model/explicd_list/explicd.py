import torch
from kltn_utils import kltn_const, kltn_utils
from torch import nn
from torch.nn import functional as F

from ..explicd_dict import utils as explicd_dict_utils
from . import utils as explicd_list_utils


class ExplicdList(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config

        # module
        self.clip_model, tokenizer = kltn_utils.build_clip_model(
            config.model.clip_model
        )

        clip_model_config = kltn_const.CLIP_MODELS[config.model.clip_model]
        visual_feature_dim = clip_model_config["visual_feature_dim"]
        embedding_dim = clip_model_config["embedding_dim"]
        num_heads = clip_model_config["num_heads"]

        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(config.num_concept, visual_feature_dim))
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = explicd_dict_utils.FFN(visual_feature_dim, visual_feature_dim * 4)
        self.layer_norm = nn.LayerNorm(visual_feature_dim)
        self.proj = nn.Linear(
            in_features=visual_feature_dim, out_features=embedding_dim, bias=False
        )
        self.cls_head = nn.Linear(
            in_features=config.num_concept,
            out_features=config.num_class,
        )

        # concept feat
        self.register_buffer(
            "concept_feat",
            explicd_list_utils.get_concept_feat(
                config.model.clip_model, config.concepts, config
            ),
        )

        # logit scale
        self.register_buffer(
            "logit_scale",
            torch.tensor(
                kltn_const.CLIP_MODELS[config.model.clip_model]["logit_scale"]
            ),
        )

        # hook
        self.visual_features = None
        visual_feature_layer = explicd_dict_utils.get_visual_feature_layer(
            self.clip_model, config.model.clip_model
        )
        visual_feature_layer.register_forward_hook(self.hook_fn)

        # grad
        explicd_dict_utils.unfreeze_visual_encoder(
            self.clip_model, config.model.clip_model
        )

    def hook_fn(self, module, input, output):
        """
        Forward hook to capture patch-level visual features
        output shape: (B, 1 + N_patches, D)
        """
        self.visual_features = output

    def get_bridge_param(self):
        clip_param_ids = {id(p) for p in self.clip_model.parameters()}

        params = [p for p in self.parameters() if id(p) not in clip_param_ids]

        return params

    def forward(self, imgs):
        batch_size = imgs.shape[0]

        self.clip_model(imgs, None)
        img_feat_map = self.visual_features[:, 1:, :]

        visual_tokens = self.visual_tokens.repeat(batch_size, 1, 1)
        agg_visual_tokens, _ = self.cross_attn(
            visual_tokens, img_feat_map, img_feat_map
        )
        agg_visual_tokens = agg_visual_tokens.mean(dim=1)
        agg_visual_tokens = F.normalize(
            self.proj(self.layer_norm(self.ffn(agg_visual_tokens))), dim=-1
        )

        concept_logits = self.logit_scale * (agg_visual_tokens @ self.concept_feat.T)
        cls_logits = self.cls_head(concept_logits)

        return cls_logits, concept_logits
