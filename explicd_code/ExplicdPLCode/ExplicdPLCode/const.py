from kltn_utils import kltn_utils

from . import concept_dataset

CP_PATH = "/kaggle/working/checkpoint"
INPUT_PATH = "/kaggle/input/datasets/tmtrnhelloworld/explicdplcode"

CLASS_AND_CONCEPT = {
    "isic2018": kltn_utils.read_json_to_dict(
        f"{INPUT_PATH}/class_concept/isic2018/class_concept.json"
    )
}

CONCEPT_DATASET_DICT = {
    "isic2018": concept_dataset.explicid_isic_dict,
    "IDRID": concept_dataset.explicid_idrid_dict,
    "BUSI": concept_dataset.explicid_busi_dict,
    "nct_crc_he": concept_dataset.explicid_nct_dict,
    "lcc": concept_dataset.explicid_lungcolon_dict,
}

CLASS_NAMES = {
    "IDRID": ["0", "1", "2", "3", "4"],
    "BUSI": ["normal", "malignant", "benign"],
    "isic2018": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"],
    "nct_crc_he": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
    "lcc": [
        "Colon adenocarcinoma",
        "Colon benign tissue",
        "Lung adenocarcinoma",
        "Lung benign tissue",
        "Lung squamous cell carcinoma",
    ],
}


CLASS2CONCEPT = {
    "isic2018": [
        [0, 0, 0, 0, 0, 0, 0],  # MEL
        [1, 1, 1, 1, 1, 1, 0],  # NV
        [2, 0, 2, 2, 2, 0, 1],  # BCC
        [3, 0, 0, 3, 3, 0, 2],  # AKIEC
        [4, 2, 1, 4, 4, 1, 3],  # BKL
        [5, 1, 1, 5, 5, 1, 0],  # DF
        [6, 3, 1, 6, 1, 2, 0],  # VASC
    ],
    "IDRID": [
        # DR0
        [0, 0, 0, 0, 0, 0, 0, 0],
        # DR1
        [1, 1, 0, 0, 1, 0, 1, 1],
        # DR2
        [2, 2, 2, 1, 1, 0, 2, 2],
        # DR3
        [3, 3, 3, 2, 4, 0, 3, 3],
        # DR4
        [3, 4, 2, 2, 4, 2, 3, 4],
    ],
    "BUSI": [
        # NORMAL (0)
        [0, 0, 0, 0, 0, 0, 0],
        # MALIGNANT (1)
        [1, 2, 1, 2, 2, 1, 1],
        # BENIGN (2)
        [0, 0, 0, 1, 1, 0, 0],
    ],
    "nct_crc_he": [
        # ADI
        [1, 3, 0, 0, 1, 0, 2],
        # BACK
        [0, 0, 4, 0, 0, 0, 0],
        # DEB
        [4, 3, 4, 2, 5, 6, 3],
        # LYM
        [2, 3, 5, 4, 0, 3, 1],
        # MUC
        [3, 3, 6, 1, 4, 4, 0],
        # MUS
        [1, 3, 1, 1, 0, 2, 2],
        # NORM
        [3, 3, 3, 2, 2, 4, 2],
        # STR
        [1, 3, 2, 1, 0, 1, 2],
        # TUM
        [3, 3, 3, 3, 3, 5, 4],
    ],
    "lcc": [
        [0, 0, 2, 2, 1, 1, 0],  # Colon adenocarcinoma
        [0, 3, 0, 0, 0, 1, 0],  # Colon benign tissue
        [1, 1, 1, 1, 0, 1, 0],  # Lung adenocarcinoma
        [1, 3, 0, 0, 0, 0, 0],  # Lung benign tissue
        [1, 2, 2, 2, 2, 0, 1],  # Lung squamous cell carcinoma
    ],
}
