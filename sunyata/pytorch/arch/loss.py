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