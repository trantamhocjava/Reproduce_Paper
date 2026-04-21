import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image


def pickle_load(path, encoding="ASCII"):
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding)


def to_png():
    datas = {
        "filenames": [],
        "data": [],
        "labels": [],
        "batch_label": [],
    }
    for path in [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
    ]:
        bdata = dict(pickle_load(f"data/{path}", encoding="bytes").items())
        datas["filenames"] += bdata[b"filenames"]
        datas["data"].append(bdata[b"data"])
        datas["labels"] += bdata[b"labels"]
        datas["batch_label"] += [bdata[b"batch_label"]] * len(bdata[b"labels"])

    datas["data"] = np.concatenate(datas["data"])
    label_names = pickle_load("data/batches.meta", encoding="bytes")[b"label_names"]
    filenames = datas["filenames"]
    data = datas["data"]
    labels = datas["labels"]
    info = []
    for filename, img, label in zip(filenames, data, labels):
        img_path = f'images/{filename.decode("ASCII")}'
        img = img.reshape(3, 32, 32).transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        if os.path.exists(img_path):
            continue
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)
        info.append(
            {
                "label": label,
                "class": label_names[label].decode("ASCII"),
                "path": img_path,
            }
        )
    pd.DataFrame(info).to_csv("datainfo.csv", index=False)


if __name__ == "__main__":
    to_png()
