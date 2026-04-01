import os

from torch.utils.data import Dataset

from .. import utils


class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, transform, config):
        self.transforms = transform

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

        img = utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class CustomConceptDataset(Dataset):
    def __init__(self, concepts, tokenizer):
        self.concepts = concepts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.concepts)

    def __getitem__(self, idx):
        concept = self.concepts[idx]

        concept_token = self.tokenizer(concept)

        return concept_token
