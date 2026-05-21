import os

import torch
from kltn_utils import kltn_utils
from torch.utils.data import DataLoader, Dataset

from .. import const


class CustomCocoDataset(Dataset):
    def __init__(self, dataset, transform, tokenizer):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.dataset[index]
        img = kltn_utils.read_img(data["image_path"])

        if self.transform is not None:
            img = self.transform(img)

        encoded = self.tokenizer(
            data["caption"],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        tokens = encoded["input_ids"].squeeze(0)

        return img, tokens

    def __len__(self):
        return len(self.dataset)


def get_file_path_from_train_val(coco_dataset_dir, img_id):
    file_path = None
    train_file_path = (
        f"{coco_dataset_dir}/train2014/train2014/COCO_train2014_{int(img_id):012d}.jpg"
    )
    val_file_path = (
        f"{coco_dataset_dir}/val2014/val2014/COCO_train2014_{int(img_id):012d}.jpg"
    )

    if os.path.exists(train_file_path):
        file_path = train_file_path
    elif os.path.exists(val_file_path):
        file_path = val_file_path

    return file_path


def coco(config):
    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    captions = kltn_utils.read_json_to_dict(
        f"{config.coco_dataset_dir}/train_caption.json"
    )

    kltn_utils.rank_zero_info_newline(f"{len(captions)} captions loaded from json ")

    coco_data = []

    for caption in captions:
        file_path = get_file_path_from_train_val(
            config.coco_dataset_dir, caption["image_id"]
        )

        if file_path is None:
            continue

        coco_data.append(
            {
                "image_id": caption["image_id"],
                "image_path": file_path,
                "caption": caption["caption"],
            }
        )

    transform, _ = kltn_utils.build_transform(config.transform_method)
    dataloader = DataLoader(
        CustomCocoDataset(coco_data, transform, tokenizer),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    img_feats = []
    txt_tokens = []

    clip_model.cuda()
    clip_model.eval()

    for image, txt_token in dataloader:
        image = image.cuda()

        img_feat = kltn_utils.get_img_feat_from_clip_model(
            clip_model, config.clip_model, image
        ).cpu()

        img_feats.append(img_feat)
        txt_tokens.append(txt_token)

    img_feat = torch.cat(img_feats, dim=0)
    txt_token = torch.cat(txt_tokens, dim=0)

    torch.save(
        {"img_feat": img_feat, "txt_token": txt_token},
        f"{const.CP_PATH}/coco.pth",
    )
