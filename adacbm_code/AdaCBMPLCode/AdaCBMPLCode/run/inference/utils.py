import torch
from kltn_utils import kltn_utils

from ...model.adacbm.adacbm import AdaCBM


def load_model(config, select_concept_data):
    model = AdaCBM(
        select_concepts_data=select_concept_data,
        config=config,
    )

    kltn_utils.rank_zero_info_newline(text="Load model ok")

    model = kltn_utils.load_state_dict_for_model(
        model=model, state_dict_path=config.best_model_path
    )

    kltn_utils.rank_zero_info_newline(text="load_state_dict ok")

    return model


def inference_model(model, img):
    model.cuda()
    img = img.cuda()

    model.eval()
    with torch.no_grad():
        label_logits, concept_logits = model(img)

    label_logits = label_logits.cpu()
    concept_logits = concept_logits.cpu()

    return label_logits, concept_logits
