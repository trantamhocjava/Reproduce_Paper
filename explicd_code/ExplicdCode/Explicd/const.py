from .concept_dataset import (
    explicid_busi_dict,
    explicid_idrid_dict,
    explicid_isic_dict,
    explicid_lungcolon_dict,
    explicid_nct_dict,
)

CLS_WEIGHT_DICT = {
    "isic2018": [1, 0.5, 1.2, 1.3, 1, 2, 2],
    "IDRID": None,
    "BUSI": None,
    "nct_crc_he": [1, 1, 0.9, 0.9, 1.1, 0.8, 1.1, 1, 0.7],
    "lcc": None,
}

CONCEPT_DATASET_DICT = {
    "isic2018": explicid_isic_dict,
    "IDRID": explicid_idrid_dict,
    "BUSI": explicid_busi_dict,
    "nct_crc_he": explicid_nct_dict,
    "lcc": explicid_lungcolon_dict,
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

DEVICE = "cuda"


LATENT_DIM = {
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": (768, 512, 12),
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K": (1024, 768, 16),
}


CONCEPT_LABEL_MAP = {
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
        # [organ_context, overall_architecture, cytologic_atypia, mitotic_activity, necrosis, mucin_features, squamous_differentiation]
        [0, 0, 2, 2, 1, 1, 0],  # Colon adenocarcinoma
        [0, 3, 0, 0, 0, 1, 0],  # Colon benign tissue
        [1, 1, 1, 1, 0, 1, 0],  # Lung adenocarcinoma
        [1, 3, 0, 0, 0, 0, 0],  # Lung benign tissue
        [1, 2, 2, 2, 2, 0, 1],  # Lung squamous cell carcinoma
    ],
}
