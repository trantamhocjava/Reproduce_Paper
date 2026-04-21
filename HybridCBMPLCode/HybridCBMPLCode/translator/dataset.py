import random
from typing import Tuple

import torch
from kltn_utils import kltn_utils
from nltk.corpus import wordnet
from torch import Tensor
from torch.utils.data import Dataset

import clip


class ConceptDataset(Dataset):
    def __init__(self, data_path: str):
        self.max_seq_len = 20

        data = torch.load(data_path, map_location="cpu", weights_only=True)
        self.embeddings = data["embedding"]
        self.embeddings = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
        self.captions = data["tokens"].squeeze().type(torch.int64)

        kltn_utils.rank_zero_info_newline(
            f"self.embeddings.shape: {self.embeddings.shape}"
        )
        kltn_utils.rank_zero_info_newline(f"self.captions.shape: {self.captions.shape}")

    def __len__(self) -> int:
        return len(self.captions)

    def tokenize(self, text: str) -> Tensor:
        return clip.tokenize(text, truncate=True).type(torch.int64)[0]

    def pad_tokens(self, tokens):
        tokens = tokens.clone()
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
        return tokens

    def synonym_replace(self, text, p=0.1) -> str:
        if random.random() <= p:
            words = text.split(" ")
            # Select a random word that has at least one synonym
            eligible_words = [word for word in words if wordnet.synsets(word)]
            if not eligible_words:
                return text  # Return the original text if no eligible words found

            word = random.choice(eligible_words)
            synonyms = [
                lem.name().replace("_", " ")
                for syn in wordnet.synsets(word)
                for lem in syn.lemmas()
                if lem.name() != word
            ]

            if synonyms:
                new_word = random.choice(synonyms)
                # Replace only the first occurrence of the word in the text
                first_occurrence_index = words.index(word)
                words[first_occurrence_index] = new_word

            text = " ".join(words)
        return text

    def random_swap_tockens(self, tokens, p=0.1):
        if random.random() <= p:
            idx1, idx2 = random.sample(range(len(tokens)), 2)
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        return tokens

    def textual_noise_injection(self, text, p=0.5, error_ratio=0.1):
        if random.random() <= p:
            characters = list(text)
            num_errors = int(
                len(characters) * error_ratio
            )  # Define how many errors to introduce
            error_type = random.choice(
                ["replace", "swap", "delete"]
            )  # Choose an error type

            for _ in range(num_errors):
                position = random.randint(
                    0, len(characters) - 1
                )  # Position for the error

                if error_type == "replace":
                    characters[position] = random.choice("abcdefghijklmnopqrstuvwxyz")
                elif error_type == "swap" and len(characters) > 1:
                    swap_position = (position + 1) % len(characters)
                    characters[position], characters[swap_position] = (
                        characters[swap_position],
                        characters[position],
                    )
                elif error_type == "delete":
                    del characters[position]

            text = "".join(characters)

        return text

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        text_or_token = self.captions[item]

        embedding = self.embeddings[item]
        tokens = self.pad_tokens(text_or_token)

        return embedding, tokens


class CocoDataset(Dataset):
    def __init__(self, data_path: str):
        self.max_seq_len = 20

        coco = torch.load(data_path, map_location="cpu", weights_only=True)
        self.embeddings = coco["embedding"]
        self.embeddings = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
        self.tokens = coco["tokens"].squeeze().type(torch.int64)
        kltn_utils.rank_zero_info_newline(
            f"self.embeddings.shape: {self.embeddings.shape}"
        )
        kltn_utils.rank_zero_info_newline(f"self.tokens.shape: {self.tokens.shape}")

    def __len__(self) -> int:
        return len(self.embeddings)

    def pad_tokens(self, tokens):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat([tokens, torch.zeros(padding, dtype=torch.int64)])
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
        return tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        embeddings = self.embeddings[item]
        clip_tokens = self.tokens[item]
        clip_tokens = self.pad_tokens(clip_tokens)

        return embeddings, clip_tokens


class MergeDataset(Dataset):
    def __init__(self, conceptPath, cocoPath):
        self.dataset1 = ConceptDataset(conceptPath)
        self.dataset2 = CocoDataset(cocoPath)

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        item = None

        if idx < len(self.dataset1):
            item = self.dataset1[idx]
        else:
            item = self.dataset2[idx - len(self.dataset1)]

        return item
