# ref: https://jimypbr.github.io/2020/03/fast-ai-lesson-10-notes-looking-inside-the-model#normalization

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    def __init__(self, hidden_dim, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(hidden_dim, 1))
        self.adds = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.register_buffer('vars', torch.ones(1, hidden_dim, 1))
        self.register_buffer('means', torch.zeros(1, hidden_dim, 1))

    def update_stats(self, x):
        # x has dims (batch_size, hidden_dim, seq_len)
        m = x.mean((0,2), keepdim=True)
        v = x.var((0,2), keepdim=True)
        self.means.lerp_(m, self.mom)
        self.vars.lerp_(v, self.mom)
        return m, v

    def forward(self, x):
        if self.training:
            with torch.no_grad(): m, v = self.update_stats(x)
        else: m,v = self.means, self.vars
        x = (x-m) / (v+self.eps).sqrt()
        return x * self.mults + self.adds


class LayerNorm1d(nn.Module):
    __constants__ = ['eps']
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mults = nn.Parameter(torch.ones(hidden_dim, 1))
        self.adds = nn.Parameter(torch.zeros(hidden_dim, 1))

    def forward(self, x):
        # x has dims (batch_size, hidden_dim, seq_len)
        m = x.mean((1,), keepdim=True)
        v = x.var ((1,), keepdim=True)
        x = (x-m) / ((v+self.eps).sqrt())
        return x*self.mults + self.adds    