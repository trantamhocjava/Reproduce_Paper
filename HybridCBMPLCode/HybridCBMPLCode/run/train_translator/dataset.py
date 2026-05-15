from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def pad_tokens(tokens, max_seq_len):
    padding = max_seq_len - tokens.shape[0]

    if padding > 0:
        result_tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
    elif padding < 0:
        result_tokens = tokens[:max_seq_len]

    return result_tokens


class ConceptDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len):
        self.max_seq_len = max_seq_len

        data = torch.load(data_path, map_location="cpu", weights_only=False)

        self.embedding = F.normalize(data["concept_feat"], dim=-1)
        self.token = data["concept_token"].squeeze().type(torch.int64)

    def __len__(self) -> int:
        return len(self.token)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        embedding = self.embedding[idx]
        token = pad_tokens(self.token[idx], self.max_seq_len)

        return embedding, token


class CocoDataset(Dataset):
    def __init__(self, data_path: str, max_seq_len):
        self.max_seq_len = max_seq_len

        data = torch.load(data_path, map_location="cpu", weights_only=False)

        self.embedding = F.normalize(data["img_feat"], dim=-1)
        self.token = data["txt_token"].squeeze().type(torch.int64)

    def __len__(self) -> int:
        return len(self.token)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        embedding = self.embedding[idx]
        token = pad_tokens(self.token[idx], self.max_seq_len)

        return embedding, token


class MergeDataset(Dataset):
    def __init__(self, conceptPath, cocoPath, max_seq_len):
        self.dataset1 = ConceptDataset(conceptPath, max_seq_len)
        self.dataset2 = CocoDataset(cocoPath, max_seq_len)

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        item = None

        if idx < len(self.dataset1):
            item = self.dataset1[idx]
        else:
            item = self.dataset2[idx - len(self.dataset1)]

        embedding, token = item
        return embedding, token
