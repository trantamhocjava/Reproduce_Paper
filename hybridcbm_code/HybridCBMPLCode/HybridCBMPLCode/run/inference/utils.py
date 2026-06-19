import torch
from kltn_utils import kltn_utils

from ...models.adacbm_hybridcbm.adacbm_hybridcbm import AdaHybridCBM
from ...models.explicd_hybridcbm.explicd_hybridcbm import ExplicdHybridCBM
from ...models.hybridcbm.hybridcbm import HybridCBM


def load_model(config, select_concept_data):

    if config.model_type == "hybrid":
        model = HybridCBM(config=config, select_concept_data=select_concept_data)
    elif config.model_type == "ada_hybrid":
        model = AdaHybridCBM(config=config, select_concept_data=select_concept_data)
    elif config.model_type == "explicd_hybrid":
        model = ExplicdHybridCBM(config=config, select_concept_data=select_concept_data)

    kltn_utils.rank_zero_info_newline(text="Load model ok")

    ckpt = torch.load(
        config.best_model_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = ckpt["state_dict"]
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict)

    kltn_utils.rank_zero_info_newline(text="load_state_dict ok")

    return model


def inference_model(model, img, model_type):
    model.cuda()
    img = img.cuda()

    model.eval()
    with torch.no_grad():
        if model_type in ("hybrid", "ada_hybrid"):
            label_logits, concept_logits, img_feat = model(img)
        else:
            label_logits, concept_logits = model(img)

    label_logits = label_logits.cpu()
    concept_logits = concept_logits.cpu()

    return label_logits, concept_logits
