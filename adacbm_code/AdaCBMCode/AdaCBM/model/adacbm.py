import torch
from torch import nn

from .. import utils
from ..const import DEVICE


def init_weight_concept(class_names, use_rand_init, concept2cls, init_val):
    init_weight = torch.zeros((len(class_names), concept2cls.shape[1]))

    if use_rand_init:
        torch.nn.init.kaiming_normal_(init_weight)
    else:
        init_weight.scatter_(0, concept2cls, init_val)

    return init_weight


def demo_first_time(a):
    print(f"demo_first_time: {a}")


class OurModule(nn.Module):
    def __init__(self, dim, num_layers=1, residual=False, use_img_norm=False):
        super(OurModule, self).__init__()
        self.residual = residual
        self.use_img_norm = use_img_norm

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.LeakyReLU())
        self.linear1 = nn.Sequential(*layers)

    def _get_image_embedding(self, original_emb):
        A = self.linear1(original_emb)
        if self.use_img_norm:
            A = A / A.norm(dim=-1, keepdim=True)
        if self.residual:
            A = A + original_emb
        return A

    def forward(self, _A):
        A = self._get_image_embedding(_A)
        return A


def get_weight_mat(asso_mat, mask):
    mat = asso_mat * mask
    return mat


class AdaCBM(nn.Module):
    def __init__(
        self,
        select_concepts_data,
        config,
    ):
        super().__init__()
        self.config = config

        self.clip_model, self.clip_preprocess, tokenizer = utils.build_clip_model(
            config.clip_model
        )
        self.clip_model = self.clip_model.to(DEVICE)
        self.preprocess_list = utils.get_list_preprocess_v2(self.clip_preprocess)

        self.select_idx = select_concepts_data["select_idx"][: config.num_concept]
        self.concept_feat = select_concepts_data["concept_feat"][self.select_idx].to(
            DEVICE
        )
        self.concept_raw = select_concepts_data["all_concepts"][self.select_idx]
        self.concept2cls = (
            torch.from_numpy(select_concepts_data["concept2cls"][self.select_idx])
            .long()
            .view(1, -1)
        )

        if select_concepts_data["init_weight"] is None:
            self.init_weight = init_weight_concept(
                config.class_names,
                config.use_rand_init,
                self.concept2cls,
                config.init_val,
            )
        else:
            self.init_weight = select_concepts_data["init_weight"]

        if config.cls_sim_prior:
            print("use cls prior")
            new_weights = []
            for concept_id in range(self.init_weight.shape[1]):
                target_class = int(torch.where(self.init_weight[:, concept_id] == 1)[0])
                new_weights.append(
                    select_concepts_data["class_sim"][target_class]
                    + self.init_weight[:, concept_id]
                )
            self.init_weight = torch.vstack(new_weights).T

        self.attention_block = OurModule(
            dim=self.concept_feat.shape[1],
            num_layers=config.num_layers,
            residual=config.residual,
            use_img_norm=config.use_img_norm,
        )
        concept2cls_scatter = torch.zeros(
            (len(config.class_names), len(self.select_idx))
        )
        concept2cls_scatter.scatter_(0, self.concept2cls, 1)
        self.mask = self.init_weight.clone().to(DEVICE)
        self.asso_mat = nn.Parameter(self.init_weight.clone())
        self.dot_product_bias = nn.Parameter(torch.zeros(len(self.select_idx)))
        self.asso_mat_bias = nn.Parameter(torch.zeros(len(config.class_names)))

        # Frozen
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def forward(self, imgs):
        self.clip_model.eval()
        with torch.no_grad():
            img_feat = self.clip_model(imgs, None)[0].float()

        mat = get_weight_mat(self.asso_mat, self.mask)
        image_embed = self.attention_block(img_feat)
        dot_product = image_embed @ self.concept_feat.t()
        dot_product = dot_product + self.dot_product_bias
        pred = dot_product @ mat.t()
        pred = pred + self.asso_mat_bias
        return pred, dot_product
