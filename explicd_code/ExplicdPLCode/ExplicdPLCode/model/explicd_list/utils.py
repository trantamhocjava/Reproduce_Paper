import torch.nn.functional as F
from kltn_utils import kltn_utils


def get_concept(config, concept):
    criteria, content = concept.split("::")

    if config.dataset_name == "isic2018":
        concept = f"the {criteria} of the lesion is {content}"
    elif config.dataset_name == "IDRID":
        concept = f"the {criteria} of the retina is {content}"
    elif config.dataset_name == "BUSI":
        concept = f"the {criteria} of the breast is {content}"
    elif config.dataset_name == "nct_crc_he":
        concept = f"the {criteria} of the tissue is {content}"
    elif config.dataset_name == "lcc":
        concept = f"the {criteria} of the tissue is {content}"

    return concept


def get_concept_feat(clip_model_name, concepts, config):
    clip_model, tokenizer = kltn_utils.build_clip_model(clip_model_name)

    concept_list = [get_concept(config, concept) for concept in concepts]

    concept_feat = kltn_utils.get_txt_feat(
        concept_list,
        clip_model,
        clip_model_name,
        tokenizer,
        config.batch_size,
    )
    concept_feat = F.normalize(concept_feat, dim=-1)

    return concept_feat
