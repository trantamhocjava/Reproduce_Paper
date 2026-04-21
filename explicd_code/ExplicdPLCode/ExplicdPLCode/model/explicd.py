import torch
from kltn_utils import kltn_utils
from pytorch_lightning.utilities import rank_zero_info
from torch import nn
from torch.nn import functional as F

from .. import const
from . import utils as model_utils


class ExpLICD(nn.Module):
    def __init__(
        self,
        concept_dict,
        config,
    ):
        super().__init__()
        self.clip_model_name = config.clip_model

        self.clip_model, tokenizer = kltn_utils.build_clip_model(self.clip_model_name)

        self.concept_keys = list(concept_dict.keys())

        rank_zero_info("START CONCEPT FEAT")
        self.concept_token_dict = model_utils.get_concept_token_dict(
            self.clip_model_name, concept_dict, config
        )
        rank_zero_info("END CONCEPT FEAT")
        self.logit_scale = torch.tensor(const.LOGIT_SCALE[self.clip_model_name])

        self.visual_features = []
        self.hook_list = []

        visual_feature_layer = model_utils.get_visual_feature_layer(
            self.clip_model, self.clip_model_name
        )
        self.hook_list.append(visual_feature_layer.register_forward_hook(self.hook_fn))

        visual_feature_dim, concept_dim, num_heads = const.LATENT_DIM[
            self.clip_model_name
        ]
        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(len(self.concept_keys), visual_feature_dim)
            )
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = model_utils.FFN(visual_feature_dim, visual_feature_dim * 4)
        self.norm = nn.LayerNorm(visual_feature_dim)
        self.proj = nn.Linear(
            in_features=visual_feature_dim, out_features=concept_dim, bias=False
        )

        self.cls_head = nn.Linear(
            in_features=model_utils.get_num_concepts(concept_dict),
            out_features=len(config.class_names),
        )

        model_utils.replace_visual_weights_for_clip_model(
            self.clip_model, self.clip_model_name
        )

        # requires_grad
        model_utils.set_up_grad_for_clip_model(self.clip_model, self.clip_model_name)

    def hook_fn(self, module, input, output):
        """
        Forward hook to capture patch-level visual features
        output shape: (B, 1 + N_patches, D)
        """
        self.visual_features.append(output)

    def forward(self, imgs):
        device = imgs.device
        self.logit_scale = self.logit_scale.to(device)
        self.concept_token_dict = model_utils.dict_to_device(
            self.concept_token_dict, device
        )

        batch_size = imgs.shape[0]

        self.visual_features.clear()
        self.clip_model(imgs, None)
        img_feat_map = self.visual_features[0][:, 1:, :]

        visual_tokens = self.visual_tokens.repeat(batch_size, 1, 1)
        agg_visual_tokens, _ = self.cross_attn(
            visual_tokens, img_feat_map, img_feat_map
        )
        agg_visual_tokens = self.proj(self.norm(self.ffn(agg_visual_tokens)))
        agg_visual_tokens = F.normalize(agg_visual_tokens, dim=-1)

        concept_logits_dict = {}
        for idx, key in enumerate(self.concept_keys):
            concept_logits_dict[key] = (
                self.logit_scale
                * agg_visual_tokens[:, idx : idx + 1, :]
                @ self.concept_token_dict[key].repeat(batch_size, 1, 1).permute(0, 2, 1)
            ).squeeze(1)

        image_logits_list = []
        for key in self.concept_keys:
            image_logits_list.append(concept_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        return cls_logits, concept_logits_dict
