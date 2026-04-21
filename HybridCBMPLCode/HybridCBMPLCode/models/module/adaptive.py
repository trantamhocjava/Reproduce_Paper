import torch
import torch.nn as nn

class AdaptiveModule(nn.Module):
    def __init__(self, dim, num_layers = 1, residual = False, use_img_norm = False):
        super().__init__()
        self.residual = residual
        self.use_img_norm = use_img_norm

        layers = []

        for i in range(num_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        
        self.linears = nn.Sequential(*layers)
    
    def _get_image_embedding(self, original_emb):
        A = self.linears(original_emb)

        if self.use_img_norm:
            A = A / (A.norm(dim=-1, keepdim=True) + 1e-8)
        if self.residual:
            A = A + original_emb

        return A
    
    def forward(self, original_emb):
        return self._get_image_embedding(original_emb)