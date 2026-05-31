from kltn_utils import kltn_utils

INPUT_PATH = "/kaggle/input/datasets/tmtrnhelloworld/adacbmplcode"


LOSS_DICT = {
    "loss": [],
    "class_loss": [],
    "concept_loss": [],
}

CLASS_AND_CONCEPT = {
    "isic2018": kltn_utils.read_json_to_dict(
        f"{INPUT_PATH}/data/isic2018/class_concept.json"
    )
}
