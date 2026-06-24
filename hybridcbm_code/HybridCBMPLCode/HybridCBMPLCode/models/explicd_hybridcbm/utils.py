import torch
from kltn_utils import kltn_utils

BIOMEDCLIP_NAMES = {
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k",
}


OPENCLIP_NAMES = {
    "ViT-B-32",
    "ViT-B-16",
    "ViT-L-14",
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
}


OPENAI_CLIP_RESNET_NAMES = {
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
}

OPENAI_CLIP_VIT_NAMES = {
    "ViT-L-14@336px",
}


def get_visual_feature_layer(model, model_name):
    if model_name in BIOMEDCLIP_NAMES:
        return model.visual.trunk.blocks[-1]

    if model_name in OPENCLIP_NAMES:
        return model.visual.transformer.resblocks[-1]

    if model_name in OPENAI_CLIP_VIT_NAMES:
        return model.visual.transformer.resblocks[-1]

    if model_name in OPENAI_CLIP_RESNET_NAMES:
        # OpenAI CLIP RN50/RN101/RN50x* dùng ModifiedResNet
        # layer4 là block visual cuối
        return model.visual.layer4[-1]

    if model_name == "flaviagiammarino/pubmed-clip-vit-base-patch32":
        return model.vision_model.encoder.layers[-1]

    if model_name == "microsoft/BiomedVLP-BioViL-T":
        return model.visual_encoder.encoder.encoder.layer4

    raise ValueError(
        f"Không hỗ trợ model_name={model_name}. "
        f"Hãy kiểm tra cấu trúc visual encoder của model."
    )


def unfreeze_visual_encoder(model, model_name):
    if model_name in BIOMEDCLIP_NAMES:
        kltn_utils.unfreeze_module(model.visual.trunk)

        # đóng băng text encoder của BiomedCLIP
        kltn_utils.freeze_module(model.text)

    elif model_name in OPENCLIP_NAMES:
        kltn_utils.unfreeze_module(model.visual)

        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)

        if hasattr(model, "text_projection"):
            model.text_projection.requires_grad_(False)

    elif model_name in OPENAI_CLIP_VIT_NAMES:
        # OpenAI CLIP ViT-L/14@336px
        kltn_utils.unfreeze_module(model.visual)

        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)

        if hasattr(model, "text_projection"):
            model.text_projection.requires_grad_(False)

    elif model_name in OPENAI_CLIP_RESNET_NAMES:
        # OpenAI CLIP ResNet: RN50, RN101, RN50x4, RN50x16, RN50x64
        kltn_utils.unfreeze_module(model.visual)

        # đóng băng text encoder
        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)

        if hasattr(model, "text_projection"):
            model.text_projection.requires_grad_(False)

    elif model_name == "flaviagiammarino/pubmed-clip-vit-base-patch32":
        kltn_utils.unfreeze_module(model.vision_model)

        kltn_utils.freeze_module(model.text_model)
        kltn_utils.freeze_module(model.text_projection)

    elif model_name == "microsoft/BiomedVLP-BioViL-T":
        kltn_utils.unfreeze_module(model.visual_encoder)

        kltn_utils.freeze_module(model.text_encoder)

    else:
        raise ValueError(
            f"Không hỗ trợ model_name={model_name}. "
            f"Hãy kiểm tra cấu trúc visual/text encoder của model."
        )


def process_img_feat_map(img_feat_map, clip_model_name):
    if clip_model_name in ("RN50", "RN101"):
        return img_feat_map.flatten(2).transpose(1, 2)

    if clip_model_name in ("RN50x4"):
        img_feat_map = torch.cat(
            [img_feat_map, img_feat_map[:, -1:, :, :]],
            dim=1,
        )
        return img_feat_map.flatten(2).transpose(1, 2)

    return img_feat_map


def forward_clip_model(img, model, model_name):
    if model_name == "flaviagiammarino/pubmed-clip-vit-base-patch32":
        return model.vision_model(pixel_values=img)

    return model(img, None)
