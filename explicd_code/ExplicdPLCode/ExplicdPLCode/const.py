from kltn_utils import kltn_utils

CP_PATH = "/kaggle/working/checkpoint"
INPUT_PATH = "/kaggle/input/datasets/tmtrnhelloworld/explicdplcode"

CLASS_AND_CONCEPT = {
    "isic2018": kltn_utils.read_json_to_dict(
        f"{INPUT_PATH}/data/isic2018/class_concept.json"
    ),
    "BUSI": kltn_utils.read_json_to_dict(f"{INPUT_PATH}/data/BUSI/class_concept.json"),
    "IDRID": kltn_utils.read_json_to_dict(
        f"{INPUT_PATH}/data/IDRID/class_concept.json"
    ),
    "LCC": kltn_utils.read_json_to_dict(f"{INPUT_PATH}/data/LCC/class_concept.json"),
    "NCT": kltn_utils.read_json_to_dict(f"{INPUT_PATH}/data/NCT/class_concept.json"),
}

LOSS_DICT = {
    "loss": [],
    "cls_loss": [],
    "concept_loss": [],
}
