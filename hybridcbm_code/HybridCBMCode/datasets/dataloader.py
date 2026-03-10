import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

class ImageDataset(Dataset):
    """
    Cung cấp (ảnh/vector, nhãn) cho quá trình train/test.
    """
    def __init__(self, img_data, labels, transform=None):
        self.img_data = img_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        data = self.img_data[idx]
        label = self.labels[idx]
        
        # Nếu data là đường dẫn file ảnh (string) và có transform thì đọc ảnh gốc
        if self.transform and isinstance(data, (str, Path)):
            img = Image.open(data).convert('RGB')
            img = self.transform(img)
            return img, label
            
        # Nếu data đã là vector (tensor) được nén sẵn, trả về luôn (Cực nhanh)
        return data, label
    

class DataBank:
    """
    Trình quản lý DataLoader thuần PyTorch. Vẫn giữ nguyên cơ chế Caching Vector siêu tốc.
    """
    def __init__(
            self, data_root, exp_root, n_shots,
            clip_encoder=None, use_img_features=True,
            batch_size=128, num_workers=0, pin_memory=False, force_compute=False
    ):
        self.exp_root = Path(exp_root)
        self.exp_root.mkdir(exist_ok=True, parents=True)

        self.data_root = Path(data_root)
        self.img_split_path = self.data_root.joinpath('splits')
        self.img_split_path.mkdir(exist_ok=True, parents=True)

        self.n_shots = n_shots
        self.use_img_features = use_img_features
        self.clip_encoder = clip_encoder
        model_name = self.clip_encoder.model_name.replace('/', '-')

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.force_compute = force_compute

        # Quản lý đường dẫn
        self.img_feat_save_dir, self.label_save_dir, self.splits = {}, {}, {}
        for mode in ['train', 'val', 'test']:
            if mode == 'train':
                self.img_feat_save_dir[mode] = self.img_split_path.joinpath(f"img_feat_{mode}_{n_shots}_{model_name}.pth")
                self.label_save_dir[mode] = self.img_split_path.joinpath(f'label_{mode}_{n_shots}.pth')
            else:
                self.img_feat_save_dir[mode] = self.img_split_path.joinpath(f"img_feat_{mode}_{model_name}.pth")
                self.label_save_dir[mode] = self.img_split_path.joinpath(f'label_{mode}.pth')
            
            # Đọc CSV file chứa danh sách ảnh
            self.splits[mode] = pd.read_csv(self.img_split_path.joinpath(f'{mode}.csv'))
            self.splits[mode]['path'] = self.splits[mode]['path'].map(lambda x: str(self.data_root.joinpath(x)))

        self.img_features, self.labels = self.prepare_img_feature()

        if self.n_shots != "all":
            self.num_images_per_class = [self.n_shots] * len(self.splits['train']['label'].unique())
        else:
            self.num_images_per_class = [len(self.splits['train'][self.splits['train']['label'] == label]) for label in self.splits['train']['label'].unique()]

        # 2. Tự động Setup Dataset ngay khi khởi tạo
        self.setup()

    def prepare_img_feature(self):
        """Ép toàn bộ ảnh qua CLIP thành vector và lưu thành file .pth để train nhanh"""
        dict_features = {}
        dict_labels = {}
        
        for mode in ['train', 'val', 'test']:
            df = self.splits[mode]
            feat_save_dir = self.img_feat_save_dir[mode]
            label_save_dir = self.label_save_dir[mode]

            if not feat_save_dir.exists() or self.force_compute:
                print(f'Đang ép vector (extract features) cho tập {mode} (Chỉ chạy 1 lần)...')
                if mode == 'train' and self.n_shots != 'all':
                    df = df.groupby('label').sample(n=self.n_shots)
                
                # Ép ảnh qua CLIP encoder
                img_features = self.clip_encoder.encode_image(df['path'].tolist(), batch_size=256) # Ép batch_size to để tiết kiệm thời gian
                label = torch.tensor(df['label'].tolist())
                
                # Lưu file .pth lại
                torch.save(img_features, feat_save_dir)
                torch.save(label, label_save_dir)
            else:
                # Nếu đã có file, chỉ việc load thẳng vào RAM siêu nhanh
                print(f'Đã tìm thấy file cache cho tập {mode}. Đang tải...')
                img_features = torch.load(feat_save_dir, weights_only=True, map_location='cpu')
                label = torch.load(label_save_dir, weights_only=True, map_location='cpu')

            dict_features[mode] = img_features
            dict_labels[mode] = label

        return dict_features, dict_labels
    
    def setup(self, stage=None):
        """Khởi tạo các Dataset chuẩn PyTorch"""
        self.datasets = {}
        for mode in ['train', 'val', 'test']:
            if self.use_img_features:
                # Train bằng vector ép sẵn (Nhanh)
                self.datasets[mode] = ImageDataset(self.img_features[mode], self.labels[mode])
            else:
                # Train bằng ảnh gốc (Chậm, dùng khi cần fine-tune luôn cả CLIP)
                raise NotImplementedError("Bạn đang bật chế độ đọc ảnh gốc. Hãy ưu tiên dùng use_img_features=True để train nhanh trên Kaggle.")
    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )