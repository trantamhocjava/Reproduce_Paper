from kltn_utils import kltn_utils
from torch.utils.data import Dataset


class CustomDatasetForBlackbox(Dataset):
    def __init__(self, dataset_dir, transforms, config):
        self.transforms = transforms

        self.file_paths, self.labels = kltn_utils.load_img_classify_data(
            dataset_dir, config.class_names
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = kltn_utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
