import os
import random
from optparse import OptionParser

import numpy as np
import torch
from torchvision.transforms import v2

from . import utils
from .const import CLASS_NAMES, DEVICE, LATENT_DIMS
from .select_algo import our_selection


def get_num_images_per_class(config):
    if config.n_shots != "all":
        num_images_per_class = [config.n_shots] * len(config.class_names)
    else:
        num_images_per_class = [
            len(os.listdir(f"{config.dataset_dir}/{class_name}"))
            for class_name in config.class_names
        ]

    return num_images_per_class


def get_img_n_shot(config):
    labels = []
    all_img_paths = []
    for class_idx, class_name in enumerate(config.class_names):
        img_names = os.listdir(f"{config.dataset_dir}/{class_name}")
        if config.n_shots != "all":
            img_names = random.sample(
                img_names, config.n_shots
            )  # random sample n shot images
        labels += [class_idx] * len(img_names)
        all_img_paths += [
            f"{config.dataset_dir}/{class_name}/{img_name}" for img_name in img_names
        ]

    return all_img_paths, labels


def prepare_img_feat(model, preprocess, latent_dim, config):
    print("compute img feat for train dataset")
    all_img_paths, labels = get_img_n_shot(config)
    img_feat = utils.prepare_img_feat(model, preprocess, all_img_paths, latent_dim)
    label = torch.tensor(labels)

    if config.use_img_norm:
        img_feat /= img_feat.norm(dim=-1, keepdim=True)

    return img_feat, label


def get_all_concepts_and_concept2cls(config):
    num_concept = sum([len(concepts) for concepts in config.class2concepts.values()])
    concept2cls = np.zeros(num_concept)
    i = 0
    all_concepts = []
    for class_name, concepts in config.class2concepts.items():
        class_idx = config.class_names.index(class_name)
        for concept in concepts:
            all_concepts.append(concept)
            concept2cls[i] = class_idx
            i += 1

    return all_concepts, concept2cls


def check_pattern(concepts, pattern):
    """
    Return a boolean array where it is true if one concept contains the pattern
    """
    return np.char.find(concepts, pattern) != -1


def check_no_cls_names(concepts, cls_names):
    res = np.ones(len(concepts), dtype=bool)
    for cls_name in cls_names:
        no_cls_name = check_pattern(concepts, cls_name) == False
        res = res & no_cls_name
    return res


def preprocess(concepts, config):
    """
    concepts: numpy array of strings of concepts

    This function checks all input concepts, remove duplication, and
    remove class names if necessary
    """
    concepts, left_idx = np.unique(concepts, return_index=True)
    if config.remove_cls_name:
        print("remove cls name")
        is_good = check_no_cls_names(concepts, config.class_names)
        concepts = concepts[is_good]
        left_idx = left_idx[is_good]
    return concepts, left_idx


def prepare_txt_feat(model, latent_dim, all_concepts, config, tokenizer):
    # TODO: it is possible to store a global text feature for all concepts
    # Here, we just be cautious to recompute it every time
    print("prepare txt feat")
    concept_feat = utils.prepare_txt_feat(
        model, all_concepts, latent_dim, tokenizer, config
    )

    if config.use_txt_norm:
        concept_feat /= concept_feat.norm(dim=-1, keepdim=True)

    return concept_feat


def select_concept(
    img_feat_train, concept_feat, concept2cls, num_images_per_class, config
):
    print("select concept")
    select_idx, selected_concept_pairs = our_selection(
        img_feat_train,
        concept_feat,
        concept2cls,
        config.num_concept,
        num_images_per_class,
        pearson_weight=config.pearson_weight,
    )

    return select_idx


def gen_init_weight_from_cls_name(model, latent_dim, concepts, config, tokenizer):
    # always use unnormalized text feature for more accurate class-concept assocation
    num_cls = len(config.class_names)
    num_concept_per_cls = config.num_concept // num_cls
    cls_name_feat = utils.prepare_txt_feat(
        model, config.class_names, latent_dim, tokenizer, config
    )
    concept_feat = utils.prepare_txt_feat(
        model, concepts, latent_dim, tokenizer, config
    )
    dis = torch.cdist(cls_name_feat, concept_feat)
    # select top k concept with smallest distanct to the class name
    _, idx = torch.topk(dis, num_concept_per_cls, largest=False)
    init_weight = torch.zeros((num_cls, config.num_concept))
    init_weight.scatter_(1, idx, 1)
    return init_weight


def gen_mask_from_img_sim(img_feat, n_shots, label):
    print("generate cls sim mask")
    num_cls = len(img_feat) // n_shots
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-7)
    img_sim = img_feat @ img_feat.T
    class_sim = torch.empty((num_cls, num_cls))
    for i, row_split in enumerate(torch.split(img_sim, n_shots, dim=0)):
        for j, col_split in enumerate(torch.split(row_split, n_shots, dim=1)):
            class_sim[label[i], label[j]] = torch.mean(col_split)

    good = class_sim >= torch.quantile(class_sim, 0.95, dim=-1)
    final_sim = torch.zeros(class_sim.shape)
    for i in range(num_cls):
        for j in range(num_cls):
            if i == j:
                final_sim[i, j] = 1
            elif good[i, j] == True:
                final_sim[i, j] = class_sim[i, j]

    return final_sim


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if config.n_shots == -1:
        config.n_shots = "all"

    config.class_names = CLASS_NAMES[config.dataset_name]
    config.class2concepts = utils.read_json(config.class2concepts_path)

    clip_model, clip_preprocess, clip_tokenizer = utils.build_clip_model(
        config.clip_model
    )
    clip_model.to(DEVICE)
    clip_preprocess = v2.Compose(utils.get_list_preprocess_v2(clip_preprocess))

    latent_dim = LATENT_DIMS[config.clip_model]

    num_images_per_class = get_num_images_per_class(config)

    img_feat, label = prepare_img_feat(clip_model, clip_preprocess, latent_dim, config)

    all_concepts, concept2cls = get_all_concepts_and_concept2cls(config)
    all_concepts, idx = preprocess(all_concepts, config)
    concept2cls = concept2cls[idx]

    concept_feat = prepare_txt_feat(
        clip_model, latent_dim, all_concepts, config, clip_tokenizer
    )

    select_idx = select_concept(
        img_feat, concept_feat, concept2cls, num_images_per_class, config
    )
    print(f"select_idx shape: {select_idx.shape}")
    print(f"first 10 select_idx: {select_idx[:10]}")

    init_weight = None
    if config.use_cls_name_init:
        init_weight = gen_init_weight_from_cls_name(
            clip_model, latent_dim, all_concepts[select_idx], config, clip_tokenizer
        )

    class_sim = None
    if config.use_cls_sim_prior and config.n_shots != "all":
        class_sim = gen_mask_from_img_sim(
            img_feat, config.n_shots, label[:: config.n_shots]
        )

    save_data = {
        "select_idx": select_idx,
        "concept_feat": concept_feat,
        "all_concepts": all_concepts,
        "concept2cls": concept2cls,
        "init_weight": init_weight,
        "class_sim": class_sim,
    }
    torch.save(save_data, config.save_data_path)

    print("done")


if __name__ == "__main__":
    print("run select_concepts")

    parser = OptionParser()
    parser.add_option(
        "--n_shots",
        dest="n_shots",
        default=-1,
        type="int",
    )
    parser.add_option(
        "--dataset_name",
        dest="dataset_name",
        type="str",
    )
    parser.add_option(
        "--dataset_dir",
        dest="dataset_dir",
        type="str",
    )
    parser.add_option(
        "--clip_model",
        dest="clip_model",
        type="str",
    )
    parser.add_option(
        "--use_img_norm",
        action="store_true",
        dest="use_img_norm",
    )
    parser.add_option(
        "--class2concepts_path",
        dest="class2concepts_path",
        type="str",
    )
    parser.add_option(
        "--remove_cls_name",
        action="store_true",
        dest="remove_cls_name",
    )
    parser.add_option(
        "--use_txt_norm",
        action="store_true",
        dest="use_txt_norm",
    )
    parser.add_option(
        "--num_concept",
        dest="num_concept",
        type="int",
    )
    parser.add_option(
        "--pearson_weight",
        dest="pearson_weight",
        type="float",
    )
    parser.add_option(
        "--use_cls_name_init",
        action="store_true",
        dest="use_cls_name_init",
    )
    parser.add_option(
        "--use_cls_sim_prior",
        action="store_true",
        dest="use_cls_sim_prior",
    )
    parser.add_option(
        "--save_data_path",
        dest="save_data_path",
        type="str",
    )
    parser.add_option("--gpu", type="str", dest="gpu", default="0")

    (cfg, args) = parser.parse_args()

    main(cfg)
