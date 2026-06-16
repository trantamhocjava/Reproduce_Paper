class ImageConceptDataset(Dataset):
    def __init__(self, dataset_dir, transform, class_names, concept2class):
        self.dataset_dir = dataset_dir
        self.transforms = transform

        self.file_paths, self.labels = kltn_utils.load_img_data_class_number(
            dataset_dir, class_names
        )

        self.class2concept = kltn_utils.build_class_concept_matrix(
            concept2class, len(class_names)
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])
        concept = self.class2concept[label]

        img = kltn_utils.read_img(file_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label, concept


train_dataset = ImageConceptDataset(
    dataset_dir=f"{dataset_dir}/{mode}",
    transform=transform,
    class_names=class_names,
    concept2class=concept2class,
)

kfold_obj = KFold(
    n_splits=config.num_fold,
    shuffle=True,
    random_state=kltn_const.SEEDING,
)


kfold_obj.split(train_dataset)
