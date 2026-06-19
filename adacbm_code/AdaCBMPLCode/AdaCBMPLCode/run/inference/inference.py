from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from kltn_utils.cbm import utils as cbm_utils
from kltn_utils.uncompress import compress

from ..train_adacbm import utils as train_adacbm_utils
from . import utils as inference_utils


def run_inference(config):
    train_adacbm_utils.setup_train(config)

    select_concept_data = torch.load(
        config.select_concepts_data_path, map_location="cpu", weights_only=False
    )

    train_transform, val_transform = kltn_utils.build_transform(config.transform)
    img, label, concept, file_paths = cbm_utils.get_label_concept(
        dataset_dir=config.dataset_dir,
        file_names=config.file_names,
        class_names=config.class_names,
        concept2class=select_concept_data["concept2class"],
        transform=val_transform,
    )

    model = inference_utils.load_model(
        config=config, select_concept_data=select_concept_data
    )
    label_logits, concept_logits = inference_utils.inference_model(model, img)

    label_pred = torch.argmax(label_logits, dim=1)
    concept_probs = torch.sigmoid(concept_logits)
    concept_pred = (concept_probs >= 0.5).long()

    save_data = {}
    for i, file_path in enumerate(file_paths):
        save_data[file_path] = {
            "label_logits": label_logits[i].tolist(),
            "label_pred": config.class_names[label_pred[i]],
            "concept_probs": concept_probs[i].tolist(),
            "concept_pred": concept_pred[i].tolist(),
            "label": label,
            "concept": concept.tolist(),
            "all_concepts": select_concept_data["concepts"],
        }

    kltn_utils.save_dict_to_json(
        data=save_data, filepath=f"{config.cp_path}/save_data.json"
    )


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    config, args = parser.parse_args()
    config = kltn_utils.read_json_to_namespace(config.arg_json)
    run_inference(config)

    compress.compress2zip(config.cp_path, config.cp_path)

    kltn_utils.rank_zero_info_newline("DONE")
