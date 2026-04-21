import pandas as pd
import torch
from kltn_utils import kltn_utils
from torch.utils.data import DataLoader, Dataset

from clip.simple_tokenizer import SimpleTokenizer

from .. import const


def tokenize(config):
    conceptNet = pd.read_json(f"{config.concept_net_dataset_dir}/conceptNet.json")
    generated = pd.read_json(f"{config.concept_net_dataset_dir}/generatedConcepts.json")
    concepts = pd.concat([conceptNet, generated]).reset_index(drop=True)

    concepts.to_json(f"{const.CP_PATH}/conceptsBank.json", index=False)

    kltn_utils.rank_zero_info_newline(f"Total concepts: {len(concepts)}")

    tokenizer = SimpleTokenizer()
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]

    all_tokens = []
    for text in concepts["concept"]:
        try:
            all_tokens.append([sot_token] + tokenizer.encode(text) + [eot_token])
        except Exception as e:
            kltn_utils.rank_zero_info_newline(f"Error: {e} for text: {text}")

    result = torch.zeros(len(all_tokens), config.context_length, dtype=torch.int)
    all_len = []
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > config.context_length:
            tokens = tokens[: config.context_length]
            tokens[-1] = eot_token
        all_len.append(tokens.__len__())
        result[i, : len(tokens)] = torch.tensor(tokens)
    all_len = torch.tensor(all_len).float()

    kltn_utils.rank_zero_info_newline("SUMMARIZE")
    kltn_utils.rank_zero_info_newline(f"Max length: {all_len.max()}")
    kltn_utils.rank_zero_info_newline(f"Min length: { all_len.min()}")
    kltn_utils.rank_zero_info_newline(f"Avg length: {all_len.mean()}")
    kltn_utils.rank_zero_info_newline(f"Std length: {all_len.std()}")

    kltn_utils.rank_zero_info_newline(
        f"Lambda Length: {int(all_len.mean() + all_len.std() * 10)}"
    )

    return result


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def concept_net(config):
    tokens = tokenize(config)
    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    dataloader = DataLoader(
        CustomDataset(tokens),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )
    all_embeddings = []
    all_tokens = []

    clip_model.to(config.device)
    for tokens in dataloader:
        tokens = tokens.to(config.device)

        embeddings = kltn_utils.get_concept_feat_from_clip_model(
            clip_model, config.clip_model, tokens
        ).cpu()

        all_embeddings.append(embeddings)
        all_tokens.append(tokens)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)

    torch.save(
        {"embedding": all_embeddings, "tokens": all_tokens},
        f"{const.CP_PATH}/concept_net.pth",
    )
