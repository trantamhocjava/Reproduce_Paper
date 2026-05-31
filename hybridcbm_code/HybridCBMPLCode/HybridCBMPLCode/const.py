from kltn_utils import kltn_utils

INPUT_PATH = "/kaggle/input/datasets/tmtrnhelloworld/hybridcbmplcode"

TOKENIZER_NAME = "openai/clip-vit-base-patch32"

LOSS_DICT = {
    "loss": [],
    "discri_loss": [],
    "ort_loss": [],
    "align_loss": [],
    "class_loss": [],
    "concept_loss": [],
}

CLASS_AND_CONCEPT = {
    "isic2018": kltn_utils.read_json_to_dict(
        f"{INPUT_PATH}/data/isic2018/class_concept.json"
    )
}


CEL_WEIGHT = {
    "isic2018": {
        "akiec": 1.8548,
        "bcc": 1.1802,
        "bkl": 0.5519,
        "df": 5.2746,
        "nv": 0.0905,
        "mel": 0.5449,
        "vasc": 4.2694,
    }
}
