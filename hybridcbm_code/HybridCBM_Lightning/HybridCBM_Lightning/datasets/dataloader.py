import os.path

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path
from PIL import Image

class ImageDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a PIL Image
    """

    def __init__(self, img, label, transform=None):
        self.img = img
        self.labels = label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.labels[idx]
        if self.transform:
            img = Image.open(img).convert('RGB')
            img = self.transform(img)
        return img, label
    
class DotProductDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a PIL Image
    """

    def __init__(self, img_feat, txt_feat, label):
        self.img_feat = img_feat
        self.txt_feat = txt_feat.t()
        self.dot_product = (img_feat @ txt_feat.t())
        self.labels = label

    def __len__(self):
        return len(self.dot_product)

    def __getitem__(self, idx):
        return self.dot_product[idx], self.labels[idx]
    
class Dataset_with_name(Dataset):
    def __init__(self, ori_dataset, names):
        if len(ori_dataset) != len(names):
            raise ValueError("Length of ori_dataset and names must be the same")
        self.names = names
        self.ori_dataset = ori_dataset
        

    def __len__(self):
        return len(self.ori_dataset)

    def __getitem__(self, idx):
        return self.ori_dataset[idx] + (str(self.names[idx]),)
    
class DataBank(L.LightningDataModule):
    def __init__(self, data_root, exp_root, n_shots, clip_encoder = None, use_img_features = True, 
                 batch_size = 64, num_workers = 4, pin_memory = True, force_compute = False):
        super().__init__()
        self.exp_root = exp_root
        self.exp_root.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(data_root)
        self.img_split_path = self.data_root.joinpath('splits')
        self.img_split_path.mkdir(exist_ok=True, parents=True)
        
        #clip config
        self.n_shots = n_shots
        self.clip_encoder = clip_encoder
        self.use_img_features = use_img_features
        model_name = self.clip_encoder.model_name if self.clip_encoder else 'None'

        # dataloader config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # image feature is costly to compute, so it will always be cached
        self.force_compute = force_compute

        self.img_feat_save_dir, self.label_save_dir, self.splits = dict(), dict(), dict()
        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                self.img_feat_save_dir[mode] = self.exp_root.joinpath(f"img_feat_{mode}_{n_shots}_{model_name.replace('/', '-')}.pth")
                self.label_save_dir[mode] = self.exp_root.joinpath(f'label_{mode}_{n_shots}.pth')
            else:
                self.img_feat_save_dir[mode] = self.exp_root.joinpath(f"img_feat_{mode}_{model_name.replace('/', '-')}.pth")
                self.label_save_dir[mode] = self.exp_root.joinpath(f'label_{mode}_{n_shots}.pth')
            
            self.splits[mode] = pd.read_csv(self.img_split_path.joinpath(f'{mode}.csv'))
            self.splits[mode]['path'] = self.splits[mode]['path'].map(lambda x: self.data_root.joinpath(x))

        self.img_features, self.labels = self.prepare_img_feature()


        if self.n_shots != "all":
            self.num_images_per_class = [self.n_shots] * len(self.splits['train']['label'].unique())
        else:
            self.num_images_per_class = [len(self.splits['train'][self.splits['train']['label'] == label]) for label in self.splits['train']['label'].unique()]
                            

    def prepare_img_feature(self):
        dict_features = {}
        dict_labels = {}
        for mode in ['train', 'val', 'test']:
            df, feat_save_dir, label_save_dir = self.splits[mode], self.img_feat_save_dir[mode], self.label_save_dir[mode]

            if not feat_save_dir.exists():
                print(f"Compute img features for {mode}")
                if mode == "train" and self.n_shots != "all":
                    df = df.groupby("label").sample(n= self.n_shots)
                img_features = self.clip_encoder.encode_image(df['path'].tolist(), batch_size=512)
                label = torch.tensor(df['label'].tolist())
                torch.save(img_features, feat_save_dir)
                torch.save(label, label_save_dir)
            else:
                img_features, label = torch.load(feat_save_dir, weights_only=True), torch.load(label_save_dir,weights_only=True)
            
            dict_features[mode] = img_features
            dict_labels[mode] = label
        
        if self.n_shots != "all":
            if len(dict_features['train']) != len(dict_labels['train']):
                raise ValueError(f"Number of training samples does not match n_shots * number of classes. Got {len(dict_features['train'])} samples, expected {self.n_shots * len(dict_labels['train'].unique())}.")
        return dict_features, dict_labels

    def random_sample_img(self, num_images_per_class = 1, mode= 'test'):
        """
        Randomly select num_images_per_class images from
        each class in the data set.
        """
        clip_name = self.clip_encoder.model_name.replace('/', '-')
        df = self.splits[mode]
        df = df.groupby('label').sample(n=num_images_per_class).reset_index(drop=True)
        dir_path = self.data_root.joinpath(f'sampled_{clip_name}_images')
        dir_path.mkdir(exist_ok=True, parents=True)

        img_save_dir = dir_path.joinpath(f'{mode}_{num_images_per_class}_images')
        img_embed_save_path = dir_path.joinpath(f'{mode}_{num_images_per_class}_embeddings')
        img_save_dir.mkdir(exist_ok=True, parents=True)
        if not img_embed_save_path.exists():
            df.to_csv(dir_path.joinpath(f'{mode}_sampled.csv'), index=False)
            for img_path in df['path'].tolist():
                img = Image.open(img_path)
                img.save(img_save_dir.joinpath(os.path.basename(img_path)))
            embeddings = self.clip_encoder.encode_image(df['path'].tolist(), batch_size=32)
            y = torch.LongTensor(df['label'].tolist())
            torch.save({
                'embeddings': embeddings,
                'labels': y
            }, img_embed_save_path)
        else:
            embeddings = torch.load(img_embed_save_path, weights_only=True, map_location='cpu')
        return embeddings
    
    def train_transform(self):
        from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, \
            RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
        
        train_trans = Compose([
            Resize(self.clip_encoder.input_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(self.clip_encoder.input_resolution),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        return train_trans
    
    def get_train_data(self, numpy=False):
        if numpy:
            return self.img_features['train'].numpy(), self.labels['train'].numpy()
        return self.img_features['train'], self.labels['train']
    
    def get_val_data(self, numpy=False):
        if numpy:
            return self.img_features['val'].numpy(), self.labels['val'].numpy()
        return self.img_features['val'], self.labels['val']
    
    def get_test_data(self, numpy=False):
        if numpy:
            return self.img_features['test'].numpy(), self.labels['test'].numpy()
        return self.img_features['test'], self.labels['test']
    
    def setup(self, stage):
        """
        Set up datasets for dataloader to load from. Depending on the need, return either:
        - (img_features, label), concept_feat will be loaded in the models
        - (the dot product between img_features and concept_feat, label)
        - if allowing grad to image, provide (image, label)
        - if allowing grad to text, compute concept_feat inside the models
        """

        self.datasets = {}
        for mode in ['train', 'val', 'test']:
            if self.use_img_features:
                self.datasets[mode] = ImageDataset(self.img_features[mode], self.labels[mode])
            else:
                if mode == 'train':
                    self.datasets[mode] = ImageDataset(self.img_features[mode], self.labels[mode],
                                                       transform=self.train_transform())
                else:
                    self.datasets[mode] = ImageDataset(self.splits[mode], self.labels[mode],
                                                       transform=self.clip_encoder.preprocess)
    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory, )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(
            Dataset_with_name(self.datasets['test'], self.splits['test']['path'].tolist()),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)
