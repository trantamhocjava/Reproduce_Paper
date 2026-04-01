import timm
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as v1
from torchvision.transforms import v2

from ..const import DEVICE, LATENT_DIM


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    for param in m.parameters():
        param.requires_grad = True


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


def get_count_value_in_dict(concept_list):
    res = 0

    for key, value in concept_list.items():
        res += len(value)

    return res


def get_preprocess_list_v2(preprocess):
    normalize = None
    center_crop = None
    for t in preprocess.transforms:
        if isinstance(t, v1.Normalize):
            normalize = v2.Normalize(mean=t.mean, std=t.std)

        if isinstance(t, v1.CenterCrop):
            center_crop = v2.CenterCrop(size=t.size)

    return [center_crop, v2.ToDtype(torch.float32, scale=True), normalize]


def build_clip_model(model_name):
    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    preprocess_list = get_preprocess_list_v2(preprocess)

    return model, preprocess_list, tokenizer


def get_visual_feature_layer(model, model_name):
    if model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        visual_feature_layer = model.visual.trunk.blocks[-1]
    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        visual_feature_layer = model.visual.transformer.resblocks[-1]

    return visual_feature_layer


def get_prefix(config, key):
    if config.dataset_name == "isic2018":
        prefix = f"this is a dermoscopic image, the {key} of the lesion is "
    elif config.dataset_name == "IDRID":
        prefix = f"this is a dermoscopic image, the {key} of the retina is "
    elif config.dataset_name == "BUSI":
        prefix = f"this is a dermoscopic image, the {key} of the breast is "
    elif config.dataset_name == "nct_crc_he":
        prefix = f"this is a dermoscopic image, the {key} of the tissue is "
    elif config.dataset_name == "lcc":
        prefix = f"this is a dermoscopic image, the {key} of the tissue is "

    return prefix


def set_up_requires_grad(model, model_name):
    if model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        for param in model.visual.trunk.parameters():
            param.requires_grad = True
        for param in model.text.parameters():
            param.requires_grad = False
    elif (
        model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        for p in model.visual.parameters():
            p.requires_grad = True

        for p in model.token_embedding.parameters():
            p.requires_grad = False

        for p in model.transformer.parameters():
            p.requires_grad = False

        for p in model.ln_final.parameters():
            p.requires_grad = False


def replace_vit_weights_for_clip_model(model, config, clip_model):
    if clip_model == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        vit = timm.create_model(
            "vit_base_patch16_224.orig_in21k",
            pretrained=True,
            num_classes=len(config.class_names),
        )
        vit.head = nn.Identity()

        model.visual.trunk.load_state_dict(vit.state_dict())


class ExpLICD(nn.Module):
    def __init__(
        self,
        concept_list,
        config,
        clip_model="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ):
        super().__init__()

        self.clip_model = clip_model

        self.model, self.preprocess_list, tokenizer = build_clip_model(clip_model)
        self.model.to(DEVICE)

        concept_keys = list(concept_list.keys())

        self.concept_token_dict = {}
        for key in concept_keys:
            prefix = get_prefix(config, key)
            attr_concept_list = concept_list[key]
            prefix_attr_concept_list = [
                prefix + concept for concept in attr_concept_list
            ]
            tmp_concept_text = tokenizer(prefix_attr_concept_list).to(DEVICE)
            self.model.eval()
            with torch.no_grad():
                _, tmp_concept_feats, _ = self.model(None, tmp_concept_text)
            self.concept_token_dict[key] = tmp_concept_feats.float()

        self.logit_scale = self.model.logit_scale.exp().detach()

        self.visual_features = []
        self.hook_list = []

        visual_feature_layer = get_visual_feature_layer(self.model, clip_model)
        layers = [visual_feature_layer]
        for layer in layers:
            self.hook_list.append(layer.register_forward_hook(self.hook_fn))

        visual_feature_dim, concept_dim, num_heads = LATENT_DIM[clip_model]
        self.visual_tokens = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(len(concept_keys), visual_feature_dim))
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = FFN(visual_feature_dim, visual_feature_dim * 4)
        self.norm = nn.LayerNorm(visual_feature_dim)
        self.proj = nn.Linear(
            in_features=visual_feature_dim, out_features=concept_dim, bias=False
        )

        in_features = get_count_value_in_dict(concept_list)
        self.cls_head = nn.Linear(
            in_features=in_features, out_features=len(config.class_names)
        )

        replace_vit_weights_for_clip_model(self.model, config, clip_model)

        # requires_grad
        set_up_requires_grad(self.model, clip_model)

    def hook_fn(self, module, input, output):
        """
        Forward hook to capture patch-level visual features
        output shape: (B, 1 + N_patches, D)
        """
        self.visual_features.append(output)

    def get_backbone_params(self):
        if (
            self.clip_model
            == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ):
            return self.model.visual.trunk.parameters()
        elif self.clip_model == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
            return self.model.visual.parameters()

    def get_bridge_params(self):
        param_list = []

        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)

        return param_list

    def forward(self, imgs):
        self.visual_features.clear()
        self.model(imgs, None)
        img_feat_map = self.visual_features[0][:, 1:, :].float()

        B, _, _ = img_feat_map.shape
        visual_tokens = self.visual_tokens.repeat(B, 1, 1)

        agg_visual_tokens, _ = self.cross_attn(
            visual_tokens, img_feat_map, img_feat_map
        )
        agg_visual_tokens = self.proj(self.norm(self.ffn(agg_visual_tokens)))

        agg_visual_tokens = F.normalize(agg_visual_tokens, dim=-1)

        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():
            image_logits_dict[key] = (
                self.logit_scale
                * agg_visual_tokens[:, idx : idx + 1, :]
                @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)
            ).squeeze(1)
            idx += 1

        image_logits_list = []
        for key in image_logits_dict.keys():
            image_logits_list.append(image_logits_dict[key])

        image_logits = torch.cat(image_logits_list, dim=-1)
        cls_logits = self.cls_head(image_logits)

        return cls_logits, image_logits_dict
