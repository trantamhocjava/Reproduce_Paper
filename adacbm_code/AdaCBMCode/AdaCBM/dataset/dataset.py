import os

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class CustomDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        transforms,
        class_names,
    ):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.class_names = class_names

        self.file_paths = []
        self.labels = []
        for class_index, class_name in enumerate(class_names):
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

        img = read_image(
            file_path,
            mode=ImageReadMode.RGB,
        )

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
