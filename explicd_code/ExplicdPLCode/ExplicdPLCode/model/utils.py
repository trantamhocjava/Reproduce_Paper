import timm
from kltn_utils import kltn_utils
from pytorch_lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CustomConceptDataset
from .train import ConceptFeatGetter


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


def set_up_grad_for_clip_model(model, model_name):
    if model_name == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        kltn_utils.unfreeze_module(model.visual.trunk)
        kltn_utils.freeze_module(model.text)

    elif model_name == "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K":
        kltn_utils.unfreeze_module(model.visual)
        kltn_utils.freeze_module(model.token_embedding)
        kltn_utils.freeze_module(model.transformer)
        kltn_utils.freeze_module(model.ln_final)


def replace_visual_weights_for_clip_model(model, clip_model):
    if clip_model == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224":
        vit = timm.create_model(
            "vit_base_patch16_224.orig_in21k",
            pretrained=True,
        )
        vit.head = nn.Identity()

        model.visual.trunk.load_state_dict(vit.state_dict())


def get_concept_token_dict(clip_model_name, concept_dict, config):
    clip_model, tokenizer = kltn_utils.build_clip_model(clip_model_name)

    concept_token_dict = {}
    for key in concept_dict.keys():
        prefix = get_prefix(config, key)
        prefix_concept_list = [prefix + concept for concept in concept_dict[key]]

        dataset = CustomConceptDataset(concepts=prefix_concept_list)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        concept_feat_getter = ConceptFeatGetter(model=clip_model, tokenizer=tokenizer)

        tester = Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
        )
        tester.test(model=concept_feat_getter, dataloaders=dataloader)

        concept_token_dict[key] = concept_feat_getter.concept_feat

    # start debug
    for idx, value in enumerate(concept_token_dict.values()):
        kltn_utils.rank_zero_info_newline(f"{idx}: {value.shape}")

    # end debug

    return concept_token_dict


def dict_to_device(dictionary, device):
    dictionary = {key: value.to(device) for key, value in dictionary.items()}
    return dictionary
