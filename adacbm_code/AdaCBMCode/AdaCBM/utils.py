import json

import clip
import numpy as np
import torch
import torch.nn as nn
import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from sklearn import metrics
from torchvision import transforms as v1
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

from .const import CLIP_MODEL_FROM_OPENAI, DEVICE


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def batchify_run_for_img(
    process_fn, data_lst, preprocess, model, res, batch_size, use_tqdm=False
):
    data_lst_len = len(data_lst)
    num_batch = np.ceil(data_lst_len / batch_size).astype(int)
    iterator = range(num_batch)
    if use_tqdm:
        iterator = tqdm.tqdm(iterator)
    for i in iterator:
        batch_data = data_lst[i * batch_size : (i + 1) * batch_size]
        batch_res = process_fn(batch_data, preprocess, model)
        res[i * batch_size : (i + 1) * batch_size] = batch_res
        del batch_res


def read_img(img_path):
    res = None

    try:
        res = read_image(
            img_path,
            mode=ImageReadMode.RGB,
        )
    except Exception:
        img = Image.open(img_path).convert("RGB")
        res = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)

    return res


def process_img(img_paths, preprocess, model):
    img_tensor = torch.cat(
        [
            preprocess(read_img(img_path)).unsqueeze(0).to(DEVICE)
            for img_path in img_paths
        ]
    )
    model.eval()
    with torch.no_grad():
        img_feat = model(img_tensor, None)[0]
    return img_feat


def prepare_img_feat(model, preprocess, img_paths, latent_dim):
    res = torch.empty((len(img_paths), latent_dim))
    batchify_run_for_img(
        process_img, img_paths, preprocess, model, res, 512, use_tqdm=True
    )
    return res


def batchify_run_for_txt(
    process_fn, data_lst, model, tokenizer, config, res, batch_size, use_tqdm=False
):
    data_lst_len = len(data_lst)
    num_batch = np.ceil(data_lst_len / batch_size).astype(int)
    iterator = range(num_batch)
    if use_tqdm:
        iterator = tqdm.tqdm(iterator)
    for i in iterator:
        batch_data = data_lst[i * batch_size : (i + 1) * batch_size]
        batch_res = process_fn(batch_data, model, tokenizer, config)
        res[i * batch_size : (i + 1) * batch_size] = batch_res
        del batch_res


def process_txt(prompts, model, tokenizer, config):
    if config.clip_model in CLIP_MODEL_FROM_OPENAI:
        token = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(DEVICE)
    elif (
        config.clip_model
        == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        token = torch.cat([tokenizer(prompt) for prompt in prompts]).to(DEVICE)

    model.eval()
    with torch.no_grad():
        txt_feat = model(None, token)[1]
    return txt_feat


def prepare_txt_feat(model, prompts, latent_dim, tokenizer, config):
    res = torch.empty((len(prompts), latent_dim))
    batchify_run_for_txt(
        process_txt, prompts, model, tokenizer, config, res, 128, use_tqdm=True
    )
    return res


def build_clip_model(clip_model_name):
    if clip_model_name in CLIP_MODEL_FROM_OPENAI:
        model, preprocess = clip.load(clip_model_name)

    tokenizer = None
    if (
        clip_model_name
        == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        model, preprocess = create_model_from_pretrained(clip_model_name)
        tokenizer = get_tokenizer(clip_model_name)

    return model, preprocess, tokenizer


def extract_v2_normalize_from_preprocess(preprocess):
    mean = None
    std = None
    for t in preprocess.transforms:
        if isinstance(t, v1.Normalize):
            mean, std = t.mean, t.std

    return v2.Normalize(mean=mean, std=std)


def extract_v2_center_crop_from_preprocess(preprocess):
    size = None
    for t in preprocess.transforms:
        if isinstance(t, v1.CenterCrop):
            size = t.size

    return v2.CenterCrop(size=size)


def extract_v2_resize_from_preprocess(preprocess):
    resize = None
    for t in preprocess.transforms:
        if isinstance(t, v1.Resize):
            size = t.size
            antialias = t.antialias
            interpolation = t.interpolation

            resize = v2.Resize(
                size=size,
                interpolation=interpolation,
                antialias=antialias,
            )

    return resize


def get_list_preprocess_v2(clip_preprocess):
    resize = extract_v2_resize_from_preprocess(clip_preprocess)
    center_crop = extract_v2_center_crop_from_preprocess(clip_preprocess)
    normalize = extract_v2_normalize_from_preprocess(clip_preprocess)

    return [resize, center_crop, v2.ToDtype(torch.float32, scale=True), normalize]


def validation(model, dataloader, criterion):
    losses_cls = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)
            cls_logits, dot_product = model(data)

            loss_cls = criterion(cls_logits, label)
            losses_cls += loss_cls.item()

            _, label_pred = torch.max(cls_logits, dim=1)

            pred_list = np.concatenate(
                (pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_list = np.concatenate(
                (gt_list, label.cpu().numpy().astype(np.uint8)), axis=0
            )

    bmac = metrics.balanced_accuracy_score(gt_list, pred_list) * 100
    acc = metrics.accuracy_score(gt_list, pred_list) * 100
    losses_cls = losses_cls / len(dataloader)

    return bmac, acc, losses_cls


def build_criterion(config):
    if config.cls_weight is None:
        criterion = nn.CrossEntropyLoss().to(DEVICE)
    else:
        lesion_weight = torch.FloatTensor(config.cls_weight).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).to(DEVICE)

    return criterion
