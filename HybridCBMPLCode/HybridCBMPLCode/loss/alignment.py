import torch
from geomloss import SamplesLoss
from torch import nn


class MaxMeanDiscrepancy(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, concept_vectors, static_vectors):
        # Compute Gaussian kernels
        concept = torch.exp(
            -1
            * (concept_vectors[:, None] - concept_vectors).pow(2).sum(2)
            / (2 * self.sigma**2)
        )
        static = torch.exp(
            -1
            * (static_vectors[:, None] - static_vectors).pow(2).sum(2)
            / (2 * self.sigma**2)
        )
        cross = torch.exp(
            -1
            * (concept_vectors[:, None] - static_vectors).pow(2).sum(2)
            / (2 * self.sigma**2)
        )

        # Compute MMD Loss
        loss = concept.mean() + static.mean() - 2 * cross.mean()
        return loss


class MeanCovarianceAlignmentLoss(nn.Module):
    def forward(self, concept_vectors, static_vectors):
        concept_mean = concept_vectors.mean(0)
        static_mean = static_vectors.mean(0)

        N1 = concept_vectors.size(0)
        N2 = static_vectors.size(0)
        concept_cov = (
            (concept_vectors - concept_mean).T
            @ (concept_vectors - concept_mean)
            / (N1 - 1)
        )
        static_cov = (
            (static_vectors - static_mean).T @ (static_vectors - static_mean) / (N2 - 1)
        )

        mean_diff = torch.norm(concept_mean - static_mean, p=2) ** 2
        cov_diff = torch.norm(concept_cov - static_cov, p="fro") ** 2

        loss = mean_diff + cov_diff
        return loss


class KLDivergenceLoss(nn.Module):
    def forward(self, concept_vectors, static_vectors):
        """
        :param concept_vectors: N1, d
        :param static_vectors: N2, d
        :return:
        """
        p_mean = torch.nn.functional.softmax(concept_vectors, dim=-1).mean(0)
        q_mean = torch.nn.functional.softmax(static_vectors, dim=-1).mean(0)
        eps = 1e-10
        p_mean = torch.clamp(p_mean, eps, 1.0)
        q_mean = torch.clamp(q_mean, eps, 1.0)

        kl_div = torch.sum(p_mean * torch.log(p_mean / q_mean))

        return kl_div


class SinkhornDistanceLoss(nn.Module):
    def __init__(self, loss="sinkhorn", p=2, blur=0.1, scaling=0.5):
        super().__init__()
        self.sinkhorn_distance = SamplesLoss(loss=loss, p=p, blur=blur, scaling=scaling)

    def forward(self, dynamic_concept, static_concept):
        """
        Warning: Sinkhorn distance is not like cosine similarity,
                 do not normalize input vectors
        :param concept_vectors: N1, d
        :param static_vectors: N2, d
        :return:
        """
        loss = self.sinkhorn_distance(dynamic_concept, static_concept)

        return loss
