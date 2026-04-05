import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


def read_img(img_path):
    res = None

    try:
        res = read_image(
            img_path,
            mode=ImageReadMode.RGB,
        )
    except Exception:
        img = Image.open(img_path).convert("RGB")
        res = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)

    return res


class CUBDataset(Dataset):
    """CUB Dataset object with caching"""

    def __init__(self, dataset_dir, config, img2attr, transforms=None):
        self.transforms = transforms
        self.img2attr = img2attr
        self.transforms = transforms

        self.file_paths = []
        self.labels = []
        for class_index, class_name in enumerate(config.class_names):
            file_paths = [
                f"{dataset_dir}/{class_name}/{i}"
                for i in os.listdir(f"{dataset_dir}/{class_name}")
            ]
            self.file_paths += file_paths
            self.labels += [class_index] * len(file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        parts = file_path.split("/")
        splitted_path = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"

        attr_label = self.img2attr.loc[splitted_path, "attribute_label"]

        return img, label, attr_label

    def __len__(self):
        return len(self.file_paths)


class AWA2Dataset(Dataset):
    def __init__(self, dataset_dir, config, img2attr, transforms=None):
        self.transforms = transforms
        self.img2attr = img2attr

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

        img = read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        concept = self.img2attr[label, :]

        return img, label, concept
