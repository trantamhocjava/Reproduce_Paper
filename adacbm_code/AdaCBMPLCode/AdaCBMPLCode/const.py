import torch
from torchvision.transforms import v2

from .gpt_concepts import concept_dataset

SEEDING = 42
CLASS2CONCEPT = {"isic2018": concept_dataset.isic2018_concept_dataset}
CP_PATH = "/kaggle/working/checkpoint"
PREPROCESS_LIST = [
    v2.Resize(size=224, interpolation="bicubic", max_size=None, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]
METRIC_MAX = ("val_c_acc_overall", "val_c_acc", "val_y_acc", "val_y_bmac")
CSV_LOGS = "csv_logs"

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
