# %% directCLR Understanding Self-Supervised Learning Dynamics without Contrastive Pairs
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# %%
x = torch.randn(10000, 16)
# %%
net = nn.Linear(16, 16).cuda()
# %%
lr = 1e-3
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0)
# %%
noise = 1.5
def augmentation(x, noise):
    x1 = x + torch.cat([torch.zeros(128, 8), torch.randn(128, 8) * noise], dim=1).cuda()
    x2 = x + torch.cat([torch.zeros(128, 8), torch.randn(128, 8) * noise], dim=1).cuda()
    return x1, x2
# %%
def infoNCE(z1, z2, temperature=0.1):
    logits = z1 @ z2.T
    logits /= temperature
    n = z1.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss

# %%
def spectrum(net):
    x = torch.randn(1000, 16).cuda()
    z = net(x)
    z = z.cpu().detach().numpy()
    z = np.transpose(z)
    c = np.cov(z)

    _, d, _ = np.linalg.svd(c)
    return d

# %%
for i in range(10000):
    # sample data
    idx = np.random.choice(10000, 128, replace=False)
    xi = x[idx].cuda()

    # apply augmentation
    x1, x2 = augmentation(xi, noise)

    # apply encoder
    z1 = net(x1)
    z2 = net(x2)

    # train
    optimizer.zero_grad()
    loss = infoNCE(z1, z2)
    loss.backward()
    optimizer.step()

    # print spectrum
    if i % 10 == 9:
        embedding_spectrum = spectrum(net)
        print("\n iteration:", i, embedding_spectrum)
