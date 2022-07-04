import torch
import torch.nn as nn
import torch.nn.functional as F

def infoNCE(z1:torch.Tensor, z2:torch.Tensor, temperature:float=1.):
    logits = torch.einsum('b s n, b t n -> b s t', z1, z2)
    logits /= temperature
    batch_size, seq_len, hidden_dim = z1.shape
    labels = torch.arange(0, seq_len, dtype=torch.long, device=z1.device).repeat(batch_size).reshape(batch_size, seq_len)
    loss = F.cross_entropy(logits, labels)
    return loss


class InfoNCE(nn.Module):
    def __init__(self, temperature:float = 1.):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        return infoNCE(z1, z2, self.temperature)
