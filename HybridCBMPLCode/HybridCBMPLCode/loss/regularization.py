import torch
from torch import nn


class L_norm(nn.Module):
    """
    sparse regulation loss for classification layer
    """

    def __init__(self, p=1, reduction="max"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, tensor, dim=-1):
        loss = tensor.norm(p=self.p, dim=dim)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "max":
            loss = loss.max()

        return loss


class CScoreDiversityLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, sim_score):
        loss = -torch.var(sim_score, dim=0)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "max":
            loss = loss.max()
        else:
            raise NotImplementedError
        return loss


class DiscriminabilityLoss(nn.Module):
    """
    encourage the concepts that are aligned with class y, but discouraging those from other classes.
    """

    def __init__(self, alpha=1, beta=0.1, relative_margin=0.85):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.similarity_margin = relative_margin

    def similarity(self, image, concept_vectors):
        if concept_vectors.ndim == 3:
            if concept_vectors.shape[1] == 1:
                concept_vectors = concept_vectors.squeeze(1)
            else:
                mu = concept_vectors.mean(dim=1)
                logvar = torch.log(
                    concept_vectors.var(dim=1) + 1e-6
                )  # Add small constant to avoid log(0)
                std = torch.exp(0.5 * logvar)  # Calculate standard deviation
                eps = torch.randn_like(std)  # Sample from standard normal distribution
                concept_vectors = mu + eps * std  # shape (N, classes)

        similarities_per_class = image @ concept_vectors.T  # shape (N, classes)
        return concept_vectors, similarities_per_class

    def forward(self, image, concept_vectors, target, classes_embeddings):
        """
        :param image: MxD matrix of image features
        :param concept_vectors: NxD matrix of similarities between patches and concepts embeddings
        :param target:
        :return:
        """
        device = classes_embeddings.device
        num_class = classes_embeddings.shape[0]

        concept_vectors = concept_vectors.reshape(
            (num_class, -1, concept_vectors.shape[-1])
        )
        concept_vectors, similarities_per_class = self.similarity(
            image, concept_vectors
        )  # shape (classes, D)

        target_one_hot = nn.functional.one_hot(target, num_classes=num_class)
        relevant = (
            -(similarities_per_class * target_one_hot).sum(-1).mean()
        )  # shape (N, classes)
        irrelevant = (similarities_per_class * (1 - target_one_hot)).sum(-1).mean() / (
            num_class - 1
        )

        # classes_similarity = concept_vectors @ classes_embeddings.T  # shape (classes, classes)
        classes_similarity = self.similarity(classes_embeddings, concept_vectors)[
            -1
        ]  # shape (classes, classes)

        mask = torch.eye(num_class, device=device).bool()
        relevant += (
            self.alpha
            * (self.similarity_margin - classes_similarity[mask]).abs().mean()
        )
        irrelevant += self.alpha * classes_similarity[~mask].mean()

        loss = relevant + self.beta * irrelevant
        return loss


class OrthogonalityLoss(nn.Module):
    """
    orthogonality loss to encourage diversity in learned vectors
    """

    def __init__(self, num_class):
        super().__init__()
        self.num_classes = num_class
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def intra_class_diversity(self, concept_vectors):
        """
        compute intra-class diversity
        :param concept_vectors: shape (P, D). num_concepts x dim_concept
        :return: intra-class diversity
        """
        # reshape to (num_classes, num_prot_per_class, dim_concept):
        num_prot_per_class = concept_vectors.shape[0] // self.num_classes
        concept_vectors = concept_vectors.reshape(
            self.num_classes, num_prot_per_class, -1
        )
        # shape of similarity matrix is (num_classes, num_prot_per_class, num_prot_per_class)
        sim_matrix = self.cosine_similarity(
            concept_vectors.unsqueeze(1), concept_vectors.unsqueeze(2)
        )

        # use upper traingle elements of similarity matrix (excluding main diagonal)
        upper_tri_mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        loss = sim_matrix.masked_select(upper_tri_mask).abs().mean()
        return loss

    def inter_bank_diversity(self, concept_vectors, static_vectors):
        """
        compute inter-bank diversity
        :param concept_vectors: shape (P1, D). num_concepts x dim_concept
        :param static_vectors: shape (P2, D). num_static_vectors x dim_concept
        :return: inter-bank diversity
        """
        # shape of similarity matrix is (P1, P2)
        concept_vectors = concept_vectors / concept_vectors.norm(dim=-1, keepdim=True)
        static_vectors = static_vectors / static_vectors.norm(dim=-1, keepdim=True)
        sim_matrix = concept_vectors @ static_vectors.T
        # The loss is the sum of the absolute values of the similarities
        # We use absolute value to ensure that both positive and negative similarities contribute to the loss
        loss = torch.abs(sim_matrix).abs().mean()
        return loss

    def forward(self, concept_vectors, static_vectors=None):
        """
        compute loss given the concept_vectors
        :param concept_vectors: shape (P, D). num_concepts x dim_concept
        :return: orthogonality loss either across each class, summed (or averaged), or across all classes
        """
        loss = self.intra_class_diversity(concept_vectors)
        if static_vectors is not None and static_vectors.shape[0] > 0:
            loss += self.inter_bank_diversity(concept_vectors, static_vectors)

        return loss
