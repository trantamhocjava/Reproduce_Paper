import os

import torch
from torch.utils.data import Dataset

from .. import utils
from ..const import CONCEPT_LABEL_MAP


class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transforms, return_concept_label, config):
        self.transforms = transforms
        self.return_concept_label = return_concept_label

        self.concept_label_map = torch.tensor(
            CONCEPT_LABEL_MAP[config.dataset_name], dtype=torch.long
        )

        self.file_paths = []
        self.labels = []
        for class_index, class_name in enumerate(config.class_names):
            file_paths = [
                f"{dataset_dir}/{class_name}/{i}"
                for i in os.listdir(f"{dataset_dir}/{class_name}")
            ]
            self.file_paths += file_paths
            self.labels += [class_index] * len(file_paths)

        print("load done")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.return_concept_label:
            return img, label, self.concept_label_map[label]
        else:
            return img, label
