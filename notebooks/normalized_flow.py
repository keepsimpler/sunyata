# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# Define the base distribution (in this case, a standard normal distribution)
base_dist = Normal(torch.zeros(2), torch.ones(2))

# Define the transformation function (in this case, a simple affine transformation)
class Affine(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, in_features))
        self.bias = nn.Parameter(torch.randn(in_features))

    def forward(self, x):
        z = torch.matmul(x, self.weight.t()) + self.bias
        log_det = torch.slogdet(self.weight)[1]
        return z, log_det

# Define the normalized flow model
class NormalizingFlow(nn.Module):
    def __init__(self, num_flows):
        super().__init__()
        self.num_flows = num_flows
        self.transforms = nn.ModuleList([Affine(2) for _ in range(num_flows)])

    def forward(self, x):
        log_det = 0
        for i in range(self.num_flows):
            x, ld = self.transforms[i](x)
            log_det += ld
        return x, log_det

# Define the negative log-likelihood loss function
def nll_loss(model, x):
    z, log_det = model(x)
    log_probs = base_dist.log_prob(z) + log_det
    return -log_probs.mean()

# Generate some example data from a 2D Gaussian distribution
data = Normal(torch.Tensor([0, 0]), torch.Tensor([1, 1])).sample((1000,))

# Initialize the model and optimizer
model = NormalizingFlow(num_flows=5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
for i in range(1000):
    optimizer.zero_grad()
    loss = nll_loss(model, data)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")

