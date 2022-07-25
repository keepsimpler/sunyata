import torch
import torch.nn as nn
import torch.nn.functional as F

def infoNCE2(z1:torch.Tensor, z2:torch.Tensor, temperature:float=1.):
    batch_size, seq_len, hidden_dim = z1.shape
    logits = torch.einsum('a n, b n -> a b', z1.reshape(batch_size,-1), z2.reshape(batch_size,-1))
    logits /= temperature
    batch_size, seq_len, hidden_dim = z1.shape
    labels = torch.arange(0, batch_size, dtype=torch.long, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

class InfoNCE2(nn.Module):
    def __init__(self, temperature:float = 1.):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        return infoNCE2(z1, z2, self.temperature)


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


def barlow_twins(z1:torch.Tensor, z2:torch.Tensor, temperature:float=1.):
    logits = torch.einsum('b s n, b s m -> b n m', z1, z2)
    logits /= temperature
    batch_size, seq_len, hidden_dim = z1.shape
    labels = torch.arange(0, hidden_dim, dtype=torch.long, device=z1.device).repeat(batch_size).reshape(batch_size, hidden_dim)
    loss = F.cross_entropy(logits, labels)
    return loss


class BarlowTwins(nn.Module):
    def __init__(self, temperature:float = 1.):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1:torch.Tensor, z2:torch.Tensor):
        return barlow_twins(z1, z2, self.temperature)


class BarlowTwinsLoss2d(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x:torch.Tensor):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch_size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        loss = on_diag + self.lambda_coeff * off_diag
        return loss


class BarlowTwinsLoss3d(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()

        # self.seq_len = seq_len
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x:torch.Tensor):
        b, n, m = x.shape
        assert n == m
        return x.flatten(start_dim=1)[:,:-1].view(-1, n-1, n+1)[:,:,1:].flatten(start_dim=1)

    def forward(self, z1, z2):
        # N x S x D, where N is the batch_size, S is the sequence length and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=(0,1))) / torch.std(z1, dim=(0,1))
        z2_norm = (z2 - torch.mean(z2, dim=(0,1))) / torch.std(z2, dim=(0,1))

        cross_corr = torch.einsum('b s n, b s m -> b n m', z1_norm, z2_norm)

        on_diag = torch.diagonal(cross_corr, dim1=1, dim2=2).add_(-1).pow_(2).sum(-1).mean()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum(-1).mean()

        loss = on_diag + self.lambda_coeff * off_diag
        return loss


# copy from https://github.com/gpleiss/temperature_scaling/blob/126a50975e/temperature_scaling.py
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    This metric divides the confidence space into several bins and measure
    the observed accuracy in each bin. The bin gaps between observed accuracy
    and bin confidence are summed up and weighted by the proportion of samples
    in each bin.
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, target):
        softmaxes = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(softmaxes, dim=-1)
        accuracies = predictions.eq(target)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            proportion_in_bin = in_bin.float().mean()
            if proportion_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (avg_confidence_in_bin - accuracy_in_bin) * proportion_in_bin

        return ece, confidences.mean(), accuracies.float().mean()