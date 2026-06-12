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


CEL_WEIGHT = {
    "isic2018": {
        "akiec": 1.8548,
        "bcc": 1.1802,
        "bkl": 0.5519,
        "df": 5.2746,
        "nv": 0.0905,
        "mel": 0.5449,
        "vasc": 4.2694,
    },
    "nct": {
        "ADI": 0.960000,
        "BACK": 0.946372,
        "DEB": 0.868307,
        "LYM": 0.864555,
        "MUC": 1.125704,
        "MUS": 0.738916,
        "NORM": 1.140685,
        "STR": 0.958466,
        "TUM": 0.698488,
    },
    "lcc": None,
    "idrid": {
        "0": 0.706797,
        "1": 4.044444,
        "2": 0.582400,
        "3": 1.086568,
        "4": 1.427451,
    },
    "busi": {
        "benign": 0.594285,
        "malignant": 1.238095,
        "normal": 1.961006,
    },
}
