import torch
import torch.nn as nn


class EBMLabelLoss(nn.Module):
    def forward(self, y_energy, y_true):
        y_true = y_true.unsqueeze(-1)
        energy_pos = y_energy.gather(dim=1, index=y_true)
        partition_estimate = -1 * y_energy
        partition_estimate = torch.logsumexp(partition_estimate, dim=1, keepdim=True)
        predL = energy_pos + partition_estimate
        loss = predL.mean()

        return loss


class EBMConceptLoss(nn.Module):
    def forward(self, c_energy, c_true):
        device = c_energy.device

        c_true = c_true.unsqueeze(-1).to(torch.int64)
        cpt_loss = torch.zeros([]).to(device)
        num_concept = c_energy.shape[1]

        for i in range(num_concept):
            energy_pos = c_energy[:, i : i + 1].gather(
                dim=2, index=c_true[:, i : i + 1]
            )
            partition_estimate = -1 * c_energy[:, i : i + 1]
            partition_estimate = torch.logsumexp(
                partition_estimate, dim=2, keepdim=True
            )
            predL = energy_pos + partition_estimate
            predL = predL.mean()
            cpt_loss += predL

        return cpt_loss
