import ast
import collections
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn import metrics
from torch import optim
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

from .const import CONCEPT_SEMANTICS, DEVICE, SELECTED_CONCEPTS


def str2obj(text):
    return ast.literal_eval(text)


def load_img2attr(config):
    if config.dataset_name == "cub":
        img2attr = pd.read_csv(f"{config.dataset_dir}/img2attr.csv")
        img2attr["attribute_label"] = (
            img2attr["attribute_label"].apply(str2obj).apply(torch.tensor)
        )
        img2attr["uncertain_attribute_label"] = (
            img2attr["uncertain_attribute_label"].apply(str2obj).apply(torch.tensor)
        )
        img2attr = img2attr.set_index("splitted_path")
    elif config.dataset_name == "awa2":
        img2attr = torch.tensor(
            np.load(f"{config.dataset_dir}/classes_attr_matrix.npy")
        )

    return img2attr


def get_loss(model, data, label, concepts, config, loss_label, loss_concept):
    out_xy, out_cy, out_c = model(data, concepts, is_train=True, use_cy=True)

    cls_loss = loss_label(out_xy, label)
    cy_loss = loss_label(out_cy, label)
    cpt_loss = loss_concept(out_c, concepts)
    loss = (
        config.cpt_weight * cpt_loss
        + config.cls_weight * cls_loss
        + config.cy_weight * cy_loss
    )

    return out_xy, out_cy, out_c, cls_loss, cy_loss, cpt_loss, loss


def cal_accuracy_per_ele(gt_list, pred_list):
    return (gt_list == pred_list).mean() * 100


def validation(model, dataloader, config, loss_label, loss_concept):
    num_concepts = next(iter(dataloader))[2].shape[1]

    sum_cls_loss = 0
    sum_cy_loss = 0
    sum_cpt_loss = 0
    sum_loss = 0

    pred_xy_list = np.zeros((0), dtype=np.uint8)
    pred_cy_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)
    pred_c_list = np.zeros((0, num_concepts), dtype=np.uint8)
    gt_c_list = np.zeros((0, num_concepts), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for data, label, concepts in dataloader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)
            concepts = concepts.long().to(DEVICE)

            out_xy, out_cy, out_c, cls_loss, cy_loss, cpt_loss, loss = get_loss(
                model, data, label, concepts, config, loss_label, loss_concept
            )

            sum_cls_loss += cls_loss
            sum_cy_loss += cy_loss
            sum_cpt_loss += cpt_loss
            sum_loss += loss

            _, label_pred_xy = torch.min(out_xy, 1)
            _, label_pred_cy = torch.min(out_cy, 1)
            _, pred_c = torch.min(out_c, 2)

            pred_xy_list = np.concatenate(
                (pred_xy_list, label_pred_xy.cpu().numpy().astype(np.uint8)), axis=0
            )
            pred_cy_list = np.concatenate(
                (pred_cy_list, label_pred_cy.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_list = np.concatenate(
                (gt_list, label.cpu().numpy().astype(np.uint8)), axis=0
            )
            pred_c_list = np.concatenate(
                (pred_c_list, pred_c.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_c_list = np.concatenate(
                (gt_c_list, concepts.cpu().numpy().astype(np.uint8)), axis=0
            )

    loss = sum_loss / len(dataloader)
    cls_loss = sum_cls_loss / len(dataloader)
    cy_loss = sum_cy_loss / len(dataloader)
    cpt_loss = sum_cpt_loss / len(dataloader)

    c_overall_acc = metrics.accuracy_score(gt_c_list, pred_c_list) * 100
    c_acc = cal_accuracy_per_ele(gt_c_list, pred_c_list)
    y_xy_acc = metrics.accuracy_score(gt_list, pred_xy_list) * 100
    y_cy_acc = metrics.accuracy_score(gt_list, pred_cy_list) * 100
    y_xy_bmac = metrics.balanced_accuracy_score(gt_list, pred_xy_list) * 100
    y_cy_bmac = metrics.balanced_accuracy_score(gt_list, pred_cy_list) * 100

    return (
        loss,
        cls_loss,
        cy_loss,
        cpt_loss,
        c_overall_acc,
        c_acc,
        y_xy_acc,
        y_cy_acc,
        y_xy_bmac,
        y_cy_bmac,
    )


def validation_no_loss(model, dataloader):
    num_concepts = next(iter(dataloader))[2].shape[1]

    pred_xy_list = np.zeros((0), dtype=np.uint8)
    pred_cy_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)
    pred_c_list = np.zeros((0, num_concepts), dtype=np.uint8)
    gt_c_list = np.zeros((0, num_concepts), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for data, label, concepts in dataloader:
            data, label = data.float().to(DEVICE), label.long().to(DEVICE)
            concepts = concepts.long().to(DEVICE)

            out_xy, out_cy, out_c = model(data, concepts, is_train=True, use_cy=True)

            _, label_pred_xy = torch.min(out_xy, 1)
            _, label_pred_cy = torch.min(out_cy, 1)
            _, pred_c = torch.min(out_c, 2)

            pred_xy_list = np.concatenate(
                (pred_xy_list, label_pred_xy.cpu().numpy().astype(np.uint8)), axis=0
            )
            pred_cy_list = np.concatenate(
                (pred_cy_list, label_pred_cy.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_list = np.concatenate(
                (gt_list, label.cpu().numpy().astype(np.uint8)), axis=0
            )
            pred_c_list = np.concatenate(
                (pred_c_list, pred_c.cpu().numpy().astype(np.uint8)), axis=0
            )
            gt_c_list = np.concatenate(
                (gt_c_list, concepts.cpu().numpy().astype(np.uint8)), axis=0
            )

    c_overall_acc = metrics.accuracy_score(gt_c_list, pred_c_list) * 100
    c_acc = cal_accuracy_per_ele(gt_c_list, pred_c_list)
    y_xy_acc = metrics.accuracy_score(gt_list, pred_xy_list) * 100
    y_cy_acc = metrics.accuracy_score(gt_list, pred_cy_list) * 100
    y_xy_bmac = metrics.balanced_accuracy_score(gt_list, pred_xy_list) * 100
    y_cy_bmac = metrics.balanced_accuracy_score(gt_list, pred_cy_list) * 100

    return (
        c_overall_acc,
        c_acc,
        y_xy_acc,
        y_cy_acc,
        y_xy_bmac,
        y_cy_bmac,
    )


def validation_gradient_infer(model, dataloader, config):
    inferer = GradientInference(model, config)

    for i, (data, y, concept) in enumerate(dataloader):
        print(f"Start batch {i + 1} / {len(dataloader)}")
        data, y = data.float().cuda(), y.long().cuda()
        concept = concept.long().cuda()

        start_time = time.time()
        energy = inferer.inference(data, y, concept)
        elapse_time = time.time() - start_time
        print(f"elapse_time: {elapse_time}")

        _, _, _, y_prob, c_prob = energy
        inferer.metrics.update(y_prob, c_prob, y, concept)

    y_acc, y_bmac, c_acc_overall, c_acc = inferer.metrics.return_metrics()

    return y_acc, y_bmac, c_acc_overall, c_acc


def build_optimizer(model, config):
    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.wd,
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            betas=(config.beta_1, config.beta_2),
            weight_decay=config.wd,
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.wd,
        )

    return optimizer


def build_scheduler(optimizer, config):
    if config.scheduler == "LinearLR":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=0.01,
            total_iters=config.epochs,
        )
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    return scheduler


def step_scheduler(scheduler, config, val_loss):
    if config.scheduler == "LinearLR":
        scheduler.step()
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler.step(val_loss)


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


def seed_everything(seed: int):
    # 1. Python random
    random.seed(seed)

    # 2. Numpy
    np.random.seed(seed)

    # 3. Torch CPU
    torch.manual_seed(seed)

    # 4. Torch GPU (1 GPU)
    torch.cuda.manual_seed(seed)

    # 5. Torch GPU (multi-GPU)
    torch.cuda.manual_seed_all(seed)

    # 6. cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 7. (PyTorch >=1.8) deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # 8. Hash seed (ít người biết nhưng Lightning có set)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transform(config, preprocess_list):
    if config.transform == "paper":
        train_transform = v2.Compose(
            [
                v2.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                v2.RandomResizedCrop(299),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )

        val_transform = v2.Compose(
            [
                v2.CenterCrop(299),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
            ]
        )
    elif config.transform == "follow_backbone":
        train_transform = v2.Compose(preprocess_list)
        val_transform = v2.Compose(preprocess_list)

    return train_transform, val_transform


class EarlyStopping:
    def __init__(self, patience=7, delta=1, want_max=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_param = None
        self.want_max = want_max

        self.delta = delta if want_max else -delta

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_param = None

    def __call__(self, scoring, model_state_dict):
        score = scoring if self.want_max else -scoring

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model_state_dict)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model_state_dict)
            self.counter = 0

    def save_checkpoint(self, model_state_dict):
        self.best_param = model_state_dict


class MetricCalculator:
    def __init__(self, n_concepts):
        self.total_results_y = torch.tensor([])
        self.total_targets_y = torch.tensor([])
        self.total_results_c = torch.empty((0, n_concepts))
        self.total_targets_c = torch.empty((0, n_concepts))
        self.n_concepts = n_concepts

    def update(self, y_prob, c_prob, y, c):
        y_prob_inf = y_prob.clone().detach()
        c_prob_inf = c_prob.clone().detach()
        _, met_cy = torch.max(y_prob_inf, 1)
        met_cy = met_cy.squeeze(-1)

        _, met_c = torch.max(c_prob_inf, 2)

        self.total_results_c = torch.cat((self.total_results_c, met_c.cpu()), 0)
        self.total_targets_c = torch.cat((self.total_targets_c, c.cpu()), 0)
        self.total_results_y = torch.cat((self.total_results_y, met_cy.cpu()), 0)
        self.total_targets_y = torch.cat((self.total_targets_y, y.cpu()), 0)

    def return_metrics(self):
        c_acc_overall = (
            metrics.accuracy_score(
                self.total_targets_c.cpu().numpy(), self.total_results_c.cpu().numpy()
            )
            * 100
        )
        y_acc = (
            metrics.accuracy_score(
                self.total_targets_y.cpu().numpy(), self.total_results_y.cpu().numpy()
            )
            * 100
        )
        y_bmac = (
            metrics.balanced_accuracy_score(
                self.total_targets_y.cpu().numpy(), self.total_results_y.cpu().numpy()
            )
            * 100
        )

        c_acc = 0
        for i in range(self.n_concepts):
            metrics_c = metrics.accuracy_score(
                self.total_targets_c[:, i].cpu().numpy(),
                self.total_results_c[:, i].cpu().numpy(),
            )
            c_acc += metrics_c

        c_acc = c_acc / self.n_concepts * 100

        return y_acc, y_bmac, c_acc_overall, c_acc

    def reset(self):
        self.total_results_y = torch.tensor([])
        self.total_targets_y = torch.tensor([])
        self.total_results_c = torch.tensor([])
        self.total_targets_c = torch.tensor([])

    def get_data(self):
        return (
            self.total_results_y,
            self.total_targets_y,
            self.total_results_c,
            self.total_targets_c,
        )


def select_concept_group(n, n_concept, concept_group_map):
    count = 0
    to_be_intervened = []
    for i in concept_group_map:
        to_be_intervened += concept_group_map[i]
        count += 1

        if count == n:
            return to_be_intervened


def generate_random_numbers(n, n_concept):

    numbers = set()
    while len(numbers) < float(n):
        numbers.add(random.randint(0, n_concept - 1))

    return list(numbers)


class GradientInference:
    # define hyperparams here.
    def __init__(self, model, config):
        super(GradientInference, self).__init__()
        self.model = model
        self.stop_criteria = EarlyStopping(
            patience=config.patience, delta=config.delta, want_max=False
        )
        self.metrics = MetricCalculator(n_concepts=config.cpt_size)
        if config.intervene_type is not None:
            self.metrics_intervene = MetricCalculator(n_concepts=config.cpt_size)

        self.n_classes = len(config.class_names)
        self.n_concepts = config.cpt_size
        self.cy_weight = config.cy_weight
        self.cls_weight = config.cls_weight
        self.cpt_weight = config.cpt_weight
        self.lr_c = config.lr_c
        self.lr_y = config.lr_y
        self.missingratio = config.missingratio

        self.intervene_type = config.intervene_type
        self.concept_group_map = config.concept_group_map
        self.max_iter = config.max_iter
        self.amp = config.amp

    def get_loss(self, x, y, c, use_cy):
        xy_en, cy_en, c_en, _, _ = self.model(x, (c, y), is_train=False, use_cy=use_cy)

        cpt_loss = torch.zeros([]).cuda()
        for i in range(self.n_concepts):
            c_en_per_con = c_en[:, i, :]
            predL = c_en_per_con.mean()
            cpt_loss += predL

        xy_loss = xy_en.mean()
        cy_loss = cy_en.mean()
        loss = (
            self.cls_weight * xy_loss
            + self.cpt_weight * cpt_loss
            + self.cy_weight * cy_loss
        )

        return loss, xy_loss, cpt_loss, cy_loss

    def run_optim(self, x, y, c, optim, scaler, use_cy):
        with torch.enable_grad():
            running = True
            counter = 0
            while running and counter < self.max_iter:
                self.model.eval()
                optim.zero_grad(set_to_none=True)

                if self.amp:
                    with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                        loss, xy_loss, cpt_loss, cy_loss = self.get_loss(
                            x, y, c, use_cy
                        )

                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss, xy_loss, cpt_loss, cy_loss = self.get_loss(x, y, c, use_cy)

                    loss.backward()
                    optim.step()

                scoring = xy_loss
                print(f"{counter +1}: {scoring}")

                self.stop_criteria(scoring, self.model.state_dict())

                if self.stop_criteria.early_stop:
                    running = False

                counter += 1

        with torch.no_grad():
            self.model.eval()
            energy = self.model(x, (c, y), is_train=False, use_cy=False)

        self.stop_criteria.reset()

        return energy

    def inference(self, x, y, c):
        batch_size = x.shape[0]
        self.model.energy_model.y_prob = nn.Parameter(
            torch.randn((batch_size, self.n_classes, 1)).cuda()
        )
        self.model.energy_model.c_prob = nn.Parameter(
            torch.randn((batch_size, self.n_concepts, 2)).cuda()
        ).cuda()

        optim_list = [
            {"params": [self.model.energy_model.c_prob], "lr": self.lr_c},
            {"params": [self.model.energy_model.y_prob], "lr": self.lr_y},
        ]
        optim = torch.optim.Adam(optim_list)
        scaler = torch.amp.GradScaler(DEVICE) if self.amp else None
        energy = self.run_optim(
            x,
            y,
            c,
            optim,
            scaler,
            use_cy=True,
        )

        return energy

    def inference_with_intervention(self, x, y, c):
        energy = None
        optim_list = [
            {"params": [self.model.energy_model.c_prob], "lr": self.lr_c},
            {"params": [self.model.energy_model.y_prob], "lr": self.lr_y},
        ]
        optim = torch.optim.Adam(optim_list)
        scaler = torch.amp.GradScaler(DEVICE) if self.amp else None
        energy = self.run_optim(
            x,
            y,
            c,
            optim,
            scaler,
            use_cy=True,
        )

        if self.intervene_type == "group":
            print("Start group intervention")
            len_group_concept = len(self.concept_group_map)
            c_prob_intervene = self.model.energy_model.c_prob.clone().detach()
            gt_in = int(len_group_concept * (1 - self.missingratio))
            gt_in_idx = torch.tensor(
                select_concept_group(gt_in, len_group_concept, self.concept_group_map)
            ).to(DEVICE)
            intervene_c = torch.nn.functional.one_hot(c.long(), num_classes=2)
            intervine_c = (intervene_c - 0.5) * 10
            c_prob_intervene[:, gt_in_idx, :] = intervine_c[:, gt_in_idx, :]
            y_prob = torch.zeros((x.size(0), self.n_classes, 1)).to(DEVICE)
            self.model.energy_model.y_prob = nn.Parameter(y_prob)
            self.model.energy_model.c_prob = nn.Parameter(c_prob_intervene)
            optim_list = [
                {"params": [self.model.energy_model.c_prob], "lr": self.lr_c},
                {"params": [self.model.energy_model.y_prob], "lr": self.lr_y},
            ]
            optim = torch.optim.Adam(optim_list)
            scaler = torch.amp.GradScaler(DEVICE) if self.amp else None

            energy_after_intv = self.run_optim(
                x,
                y,
                c,
                optim,
                scaler,
                use_cy=True,
            )
        elif self.intervene_type == "individual":
            print("Start individual intervention")
            c_prob_intervene = self.model.energy_model.c_prob.clone().detach()
            all_length = c.shape[-1]
            gt_in = all_length * (1 - self.missingratio)
            gt_in_idx = torch.tensor(generate_random_numbers(gt_in, all_length)).to(
                DEVICE
            )
            intervene_c = torch.nn.functional.one_hot(c.long(), num_classes=2)
            intervine_c = (intervene_c - 0.5) * 10
            c_prob_intervene[:, gt_in_idx, :] = intervine_c[:, gt_in_idx, :]
            y_prob = torch.zeros((x.size(0), self.n_classes, 1)).to(DEVICE)
            self.model.energy_model.y_prob = nn.Parameter(y_prob)
            self.model.energy_model.c_prob = nn.Parameter(c_prob_intervene)
            optim_list = [
                {"params": [self.model.energy_model.c_prob], "lr": self.lr_c},
                {"params": [self.model.energy_model.y_prob], "lr": self.lr_y},
            ]
            optim = torch.optim.Adam(optim_list)
            scaler = torch.amp.GradScaler(DEVICE) if self.amp else None
            energy_after_intv = self.run_optim(
                x,
                y,
                c,
                optim,
                scaler,
                use_cy=True,
            )

        return energy_after_intv, energy


def get_concept_group_map(config):
    concept_group_map = collections.defaultdict(list)
    for i, concept_name in enumerate(
        list(
            np.array(CONCEPT_SEMANTICS[config.dataset_name])[
                SELECTED_CONCEPTS[config.dataset_name]
            ]
        )
    ):
        group = concept_name[: concept_name.find("::")]
        concept_group_map[group].append(i)

    return concept_group_map
