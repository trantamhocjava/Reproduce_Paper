import os

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from .. import utils


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset[idx][1]  # [0]

    def __iter__(self):
        idx = (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )
        return idx

    def __len__(self):
        return self.num_samples


class CUBDataset(Dataset):
    def __init__(
        self, dataset_dir, transforms, use_attr, config, img2attr, uncertain_label=False
    ):
        self.transforms = transforms
        self.use_attr = use_attr
        self.img2attr = img2attr
        self.uncertain_label = uncertain_label

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

        if self.use_attr:
            parts = file_path.split("/")
            splitted_path = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"

            if self.uncertain_label:
                attr_label = self.img2attr.loc[
                    splitted_path, "uncertain_attribute_label"
                ]

            else:
                attr_label = self.img2attr.loc[splitted_path, "attribute_label"]

            return img, label, attr_label
        else:
            return img, label


class AWA2Dataset(Dataset):
    def __init__(
        self, dataset_dir, transforms, use_attr, config, img2attr, uncertain_label=False
    ):
        self.transforms = transforms
        self.use_attr = use_attr
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

        img = utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.use_attr:
            concept = self.img2attr[label, :]

            return img, label, concept
        else:
            return img, label
