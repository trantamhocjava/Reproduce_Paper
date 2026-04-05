import timm
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from torch import nn
from torchvision.transforms import v2


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


def get_num_concepts(concept_list):
    res = 0

    for key, value in concept_list.items():
        res += len(value)

    return res


def get_preprocess_list_v2(preprocess):
    """Transform preprocess from v1 to v2 to speed up training

    Args:
        preprocess (_type_): preprocess got from open_clip.create_model_from_pretrained
        allow model_name:
        - `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
        - `hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K`

        architecture like:
        - Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
        - CenterCrop(size=(224, 224))
        - function _convert_to_rgb
        - ToTensor()
        - Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    Returns:
        list: list of preprocess v2
    """
    resize_v1 = preprocess.transforms[0]
    resize = v2.Resize(
        size=resize_v1.size,
        interpolation=resize_v1.interpolation,
        max_size=resize_v1.max_size,
        antialias=resize_v1.antialias,
    )

    center_crop_v1 = preprocess.transforms[1]
    center_crop = v2.CenterCrop(size=center_crop_v1.size)

    normalize_v1 = preprocess.transforms[-1]
    normalize = v2.Normalize(mean=normalize_v1.mean, std=normalize_v1.std)

    return [resize, center_crop, v2.ToDtype(torch.float32, scale=True), normalize]


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


def set_up_grad_for_clip_model(model, model_name):
    if model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        unfreeze_module(model.visual.trunk)
        freeze_module(model.text)
    elif (
        model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        unfreeze_module(model.visual)
        freeze_module(model.token_embedding)
        freeze_module(model.transformer)
        freeze_module(model.ln_final)


def replace_visual_weights_for_clip_model(model, clip_model):
    if clip_model == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        vit = timm.create_model(
            "vit_base_patch16_224.orig_in21k",
            pretrained=True,
        )
        vit.head = nn.Identity()

        model.visual.trunk.load_state_dict(vit.state_dict())
