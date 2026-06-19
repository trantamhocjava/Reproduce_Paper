import os
from optparse import OptionParser

import torch
from kltn_utils import kltn_utils
from kltn_utils.uncompress import compress


def demo_run_commit(config):
    os.makedirs(config.cp_path, exist_ok=True)

    img = get_img(dataset_dir=config.dataset_dir)

    kltn_utils.rank_zero_info_newline(f"img.shape: {img.shape}")

    clip_model, _ = kltn_utils.build_clip_model(clip_model_name=config.clip_model)

    img_feat = kltn_utils.get_img_feat_from_clip_model(
        clip_model, config.clip_model, img
    )

    kltn_utils.rank_zero_info_newline(f"img_feat.shape: {img_feat.shape}")

    torch.save(
        {
            "img_feat": img_feat,
            "img_feat_short": img_feat[:5],
        },
        f"{config.cp_path}/result.pth",
    )
    torch.save(
        {
            "img_feat": img_feat,
            "img_feat_short": img_feat[:5],
        },
        f"{config.cp_path}/result_1.pth",
    )


def get_img(dataset_dir):
    img_paths = [f"{dataset_dir}/{item}" for item in os.listdir(dataset_dir)]
    img = []
    transform, _ = kltn_utils.build_transform(transform_method="uniform")
    for img_path in img_paths:
        img_item = kltn_utils.read_img(img_path)
        img_item = transform(img_item)
        img.append(img_item)

    img = torch.stack(img, dim=0)

    return img


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option(
        "--arg_json",
        type="str",
        dest="arg_json",
    )

    config, args = parser.parse_args()
    config = kltn_utils.read_json_to_namespace(config.arg_json)
    demo_run_commit(config)

    compress.compress2zip(config.cp_path, config.cp_path)

    kltn_utils.rank_zero_info_newline("DONE")
