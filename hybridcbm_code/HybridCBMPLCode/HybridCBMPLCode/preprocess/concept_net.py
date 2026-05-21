import torch
from kltn_utils import kltn_utils
from torch.utils.data import DataLoader, TensorDataset

from .. import const


def tokenize_concepts(config, tokenizer):
    conceptNet = kltn_utils.read_json_to_dict(
        f"{config.concept_net_dataset_dir}/conceptNet.json"
    )
    generated = kltn_utils.read_json_to_dict(
        f"{config.concept_net_dataset_dir}/generatedConcepts.json"
    )

    concepts = conceptNet["concept"] + generated["concepts"]
    concepts = [str(concept) for concept in concepts]

    concept_token = tokenizer(
        concepts,
        context_length=config.context_length,
    ).long()

    return concept_token


def concept_net(config):
    clip_model, tokenizer = kltn_utils.build_clip_model(config.clip_model)

    concept_token = tokenize_concepts(config, tokenizer)

    dataloader = DataLoader(
        TensorDataset(concept_token),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    concept_feats = []
    concept_tokens = []

    clip_model.cuda()
    clip_model.eval()
    for concept_token in dataloader:
        concept_token = concept_token.cuda()

        concept_feat = kltn_utils.get_concept_feat_from_clip_model(
            clip_model, config.clip_model, concept_token
        ).cpu()

        concept_feats.append(concept_feat)
        concept_tokens.append(concept_token)

    concept_feat = torch.cat(concept_feats, dim=0)
    concept_token = torch.cat(concept_tokens, dim=0)

    torch.save(
        {"concept_feat": concept_feat, "concept_token": concept_token},
        f"{const.CP_PATH}/concept_net.pth",
    )
