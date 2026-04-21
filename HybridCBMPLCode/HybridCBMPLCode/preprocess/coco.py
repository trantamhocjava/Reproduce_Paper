import os

import pandas as pd
import torch
from kltn_utils import kltn_const, kltn_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from clip import clip

from .. import const


class CustomCocoDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.tokenize = clip.tokenize

    def __getitem__(self, index):
        data = self.data[index]
        img = kltn_utils.read_img(data["image"])

        if self.transform is not None:
            img = self.transform(img)

        tokens = self.tokenize(data["caption"], truncate=True)[0]

        return img, tokens

    def __len__(self):
        return len(self.data)


def get_file_path_from_train_val(config, img_id):
    train_file_path = f"{config.coco_dataset_dir}/train2014/train2014/COCO_train2014_{int(img_id):012d}.jpg"
    val_file_path = f"{config.coco_dataset_dir}/val2014/val2014/COCO_train2014_{int(img_id):012d}.jpg"

    if os.path.exists(train_file_path):
        file_path = train_file_path
    elif os.path.exists(val_file_path):
        file_path = val_file_path
    else:
        file_path = None

    return file_path


def coco(config):
    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    captions = kltn_utils.read_json_to_dict(
        f"{config.coco_dataset_dir}/train_caption.json"
    )

    kltn_utils.rank_zero_info_newline(f"{len(captions)} captions loaded from json ")

    data = []

    for d in captions:
        img_id = d["image_id"]
        caption = d["caption"]
        file_path = get_file_path_from_train_val(config, img_id)

        if file_path is None:
            continue

        data.append({"image_id": img_id, "image": file_path, "caption": caption})

    data = pd.DataFrame(data)
    data.to_json(f"{const.CP_PATH}/coco_data.json", index=False)

    dataloader = DataLoader(
        CustomCocoDataset(
            data.to_dict(orient="records"), v2.Compose(kltn_const.PREPROCESS_LIST)
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    all_embeddings = []
    all_tokens = []

    clip_model.to(config.device)
    for image, tokens in dataloader:
        image = image.to(config.device)

        embedding = kltn_utils.get_img_feat_from_clip_model(
            clip_model, config.clip_model, image
        ).cpu()

        all_embeddings.append(embedding)
        all_tokens.append(tokens)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)

    torch.save(
        {"embedding": all_embeddings, "tokens": all_tokens},
        f"{const.CP_PATH}/coco.pth",
    )
