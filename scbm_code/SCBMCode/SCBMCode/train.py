import numpy as np
import torch
from sklearn import metrics

from . import const


class MetricCalculator:
    def __init__(self, n_concepts):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
        self.c_pred = torch.empty((0, n_concepts))
        self.c_true = torch.empty((0, n_concepts))
        self.n_concepts = n_concepts

    def update(self, y_logit, c_logit, y, c):
        y_logit = y_logit.clone().detach()
        c_logit = c_logit.clone().detach()

        _, y_pred = torch.max(y_logit, 1)

        c_pred = c_logit > 0.5

        self.c_pred = torch.cat((self.c_pred, c_pred.cpu()), 0)
        self.c_true = torch.cat((self.c_true, c.cpu()), 0)
        self.y_pred = torch.cat((self.y_pred, y_pred.cpu()), 0)
        self.y_true = torch.cat((self.y_true, y.cpu()), 0)

    def return_metrics(self):
        c_acc_overall = self.get_overall_concept_accuracy(
            self.c_true.cpu().numpy(), self.c_pred.cpu().numpy()
        )
        c_acc = self.get_concept_accuracy(
            self.c_true.cpu().numpy(), self.c_pred.cpu().numpy()
        )

        y_acc = (
            metrics.accuracy_score(self.y_true.cpu().numpy(), self.y_pred.cpu().numpy())
            * 100
        )
        y_bmac = (
            metrics.balanced_accuracy_score(
                self.y_true.cpu().numpy(), self.y_pred.cpu().numpy()
            )
            * 100
        )

        return {
            "c_acc_overall": c_acc_overall,
            "c_acc": c_acc,
            "y_acc": y_acc,
            "y_bmac": y_bmac,
        }

    def reset(self):
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
        self.c_pred = torch.tensor([])
        self.c_true = torch.tensor([])

    def get_concept_accuracy(self, c_true, c_pred):
        return (c_true == c_pred).mean() * 100

    def get_overall_concept_accuracy(self, c_true, c_pred):
        return np.mean(np.all(c_true == c_pred, axis=1)) * 100


class LossCalculator:
    def __init__(self, n_batchs):
        self.target_loss = 0
        self.concepts_loss = 0
        self.total_loss = 0
        self.n_batchs = n_batchs

    def update(self, target_loss, concepts_loss, total_loss):
        self.target_loss += target_loss.item()
        self.concepts_loss += concepts_loss.item()
        self.total_loss += total_loss.item()

    def return_metrics(self):
        target_loss = self.target_loss / self.n_batchs
        concepts_loss = self.concepts_loss / self.n_batchs
        total_loss = self.total_loss / self.n_batchs

        return {
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
        }

    def reset(self):
        self.target_loss = 0
        self.concepts_loss = 0
        self.total_loss = 0


class CBMTrain:
    def __init__(
        self,
        trainLoader,
        valLoader,
        model,
        optimizer,
        scaler,
        mode,
        config,
        loss_fn,
        epoch,
    ):
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.mode = mode
        self.config = config
        self.loss_fn = loss_fn
        self.epoch = epoch

    def train_one_epoch(self):
        metric = MetricCalculator(self.config.num_concepts)
        loss_return = LossCalculator(len(self.trainLoader))

        self.model.train()

        if self.config.training_mode in ("sequential", "independent"):
            if self.mode == "c":
                self.model.head.eval()
            elif self.mode == "t":
                self.model.encoder.eval()

        for data, label, concept in self.trainLoader:
            data = data.float().cuda()
            label = label.long().cuda()
            concept = concept.long().cuda()

            self.optimizer.zero_grad(set_to_none=True)

            if self.config.amp:
                with torch.autocast(device_type=const.DEVICE, dtype=torch.float16):
                    (
                        loss,
                        concepts_pred_probs,
                        target_pred_logits,
                        target_loss,
                        concepts_loss,
                        total_loss,
                    ) = self.get_loss(data, concept, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                (
                    loss,
                    concepts_pred_probs,
                    target_pred_logits,
                    target_loss,
                    concepts_loss,
                    total_loss,
                ) = self.get_loss(data, concept, label)

                loss.backward()
                self.optimizer.step()

            # Update loss and metric
            loss_return.update(target_loss, concepts_loss, total_loss)
            metric.update(target_pred_logits, concepts_pred_probs, label, concept)

        return loss_return.return_metrics(), metric.return_metrics()

    def validate_one_epoch(self):
        metric = MetricCalculator(self.config.num_concepts)
        loss_return = LossCalculator(len(self.valLoader))

        self.model.eval()
        with torch.no_grad():
            for data, label, concept in self.valLoader:
                data = data.float().cuda()
                label = label.long().cuda()
                concept = concept.long().cuda()

                (
                    concepts_pred_probs,
                    concepts_pred_logits,
                    target_pred_logits,
                    concepts_hard,
                ) = self.model(data, self.epoch, validation=True)

                if self.config.concept_learning == "autoregressive":
                    concepts_pred_probs = torch.mean(concepts_pred_probs, dim=-1)

                target_loss, concepts_loss, total_loss = self.loss_fn(
                    concepts_pred_logits, concept, target_pred_logits, label
                )

                # Update loss and metric
                loss_return.update(target_loss, concepts_loss, total_loss)
                metric.update(target_pred_logits, concepts_pred_probs, label, concept)

        return loss_return.return_metrics(), metric.return_metrics()

    def get_loss(self, data, concept, label):
        # Forward pass
        concepts_pred_probs, concepts_pred_logits, target_pred_logits, concepts_hard = (
            self.forward_cbm(data, concept)
        )

        # Compute the loss
        target_loss, concepts_loss, total_loss = self.loss_fn(
            concepts_pred_logits, concept, target_pred_logits, label
        )

        loss = None
        if self.mode == "j":
            loss = total_loss
        elif self.mode == "c":
            loss = concepts_loss
        else:
            loss = target_loss

        return (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def forward_cbm(self, data, concept):
        if self.config.training_mode == "independent" and self.mode == "t":
            (
                concepts_pred_probs,
                concepts_pred_logits,
                target_pred_logits,
                concepts_hard,
            ) = self.model(data, self.epoch, concept)
        elif self.config.concept_learning == "autoregressive" and self.mode == "c":
            (
                concepts_pred_probs,
                concepts_pred_logits,
                target_pred_logits,
                concepts_hard,
            ) = self.model(data, self.epoch, concepts_train_ar=concept)
        else:
            (
                concepts_pred_probs,
                concepts_pred_logits,
                target_pred_logits,
                concepts_hard,
            ) = self.model(data, self.epoch)

        return (
            concepts_pred_probs,
            concepts_pred_logits,
            target_pred_logits,
            concepts_hard,
        )


class SCBMTrain:
    def __init__(
        self,
        trainLoader,
        valLoader,
        model,
        optimizer,
        scaler,
        mode,
        config,
        loss_fn,
        epoch,
    ):
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.mode = mode
        self.config = config
        self.loss_fn = loss_fn
        self.epoch = epoch

    def train_one_epoch(self):
        metric = MetricCalculator(self.config.num_concepts)
        loss_return = LossCalculator(len(self.trainLoader))

        self.model.train()

        if (
            self.config.training_mode == "sequential"
            or self.config.training_mode == "independent"
        ):
            if self.mode == "c":
                self.model.head.eval()
            elif self.mode == "t":
                self.model.encoder.eval()

        for data, label, concept in self.trainLoader:
            data = data.float().cuda()
            label = label.long().cuda()
            concept = concept.long().cuda()

            # Backward pass depends on the training mode of the model
            self.optimizer.zero_grad(set_to_none=True)

            if self.config.amp:
                with torch.autocast(device_type=const.DEVICE, dtype=torch.float16):
                    (
                        loss,
                        concepts_pred_probs,
                        target_pred_logits,
                        target_loss,
                        concepts_loss,
                        total_loss,
                    ) = self.get_loss(data, concept, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                (
                    loss,
                    concepts_pred_probs,
                    target_pred_logits,
                    target_loss,
                    concepts_loss,
                    total_loss,
                ) = self.get_loss(data, concept, label)

                loss.backward()
                self.optimizer.step()

            # Update loss and metric
            loss_return.update(target_loss, concepts_loss, total_loss)
            metric.update(target_pred_logits, concepts_pred_probs, label, concept)

        return loss_return.return_metrics(), metric.return_metrics()

    def get_loss(self, data, concept, label):
        # Forward pass
        concepts_mcmc_probs, concepts_mcmc_logits, triang_cov, target_pred_logits = (
            self.model(data, self.epoch, c_true=concept)
        )

        # Compute the loss
        target_loss, concepts_loss, prec_loss, total_loss = self.loss_fn(
            concepts_mcmc_logits,
            concept,
            target_pred_logits,
            label,
            triang_cov,
        )

        concepts_pred_probs = concepts_mcmc_probs.mean(-1)

        loss = None
        if self.mode == "j":
            loss = total_loss
        elif self.mode == "c":
            loss = concepts_loss + prec_loss
        else:
            loss = target_loss

        return (
            loss,
            concepts_pred_probs,
            target_pred_logits,
            target_loss,
            concepts_loss,
            total_loss,
        )

    def validate_one_epoch(self):
        metric = MetricCalculator(self.config.num_concepts)
        loss_return = LossCalculator(len(self.valLoader))

        self.model.eval()
        with torch.no_grad():
            for data, label, concept in self.valLoader:
                data = data.float().cuda()
                label = label.long().cuda()
                concept = concept.long().cuda()

                (
                    concepts_mcmc_probs,
                    concepts_mcmc_logits,
                    triang_cov,
                    target_pred_logits,
                ) = self.model(data, self.epoch, validation=True, c_true=concept)

                target_loss, concepts_loss, prec_loss, total_loss = self.loss_fn(
                    concepts_mcmc_logits,
                    concept,
                    target_pred_logits,
                    label,
                    triang_cov,
                )

                concepts_pred_probs = concepts_mcmc_probs.mean(-1)

                # Update loss and metric
                loss_return.update(target_loss, concepts_loss, total_loss)
                metric.update(target_pred_logits, concepts_pred_probs, label, concept)

        return loss_return.return_metrics(), metric.return_metrics()
