from .gpt_concepts import concept_dataset

CLASS2CONCEPT = {"isic2018": concept_dataset.isic2018_concept_dataset}
CP_PATH = "/kaggle/working/checkpoint"
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
