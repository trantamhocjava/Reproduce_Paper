import os

from kltn_utils import kltn_utils
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_dir, transforms, class2concept, config):
        self.transforms = transforms
        self.class2concept = class2concept

        self.file_paths = []
        self.labels = []
        for class_index, class_name in enumerate(config.class_names):
            file_paths = [
                f"{dataset_dir}/{class_name}/{i}"
                for i in os.listdir(f"{dataset_dir}/{class_name}")
            ]
            self.file_paths += file_paths
            self.labels += [class_index] * len(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = kltn_utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        concept = self.class2concept[label]

        return img, label, concept


class CustomDatasetForBlackbox(Dataset):
    def __init__(self, dataset_dir, transforms, config):
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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = kltn_utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
