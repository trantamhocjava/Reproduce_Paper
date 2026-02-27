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

DEVICE = "cuda"

LATENT_DIMS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "RN50": 1024,
    "RN101": 512,
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 512,
}

CLIP_MODEL_FROM_OPENAI = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"]


CLS_WEIGHT_DICT = {
    "isic2018": [1, 0.5, 1.2, 1.3, 1, 2, 2],
    "IDRID": None,
    "BUSI": None,
    "nct_crc_he": [1, 1, 0.9, 0.9, 1.1, 0.8, 1.1, 1, 0.7],
    "lcc": None,
}
