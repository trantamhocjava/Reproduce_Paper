import os


def get_num_images_per_class(config):
    num_images_per_class = [
        len(os.listdir(f"{config.dataset_dir}/train/{class_name}"))
        for class_name in config.class_names
    ]

    return num_images_per_class
