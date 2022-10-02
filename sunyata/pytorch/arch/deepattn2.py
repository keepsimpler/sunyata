
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from sunyata.pytorch.arch.base import BaseCfg, BaseModule, Block, LayerScaler


class Attn(nn.Module):
    def __init__(self, hidden_dim:int, temperature: float=1., init_scale: float=1., query_idx:int=-1):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
#             nn.Linear(hidden_dim, hidden_dim)
            LayerScaler(hidden_dim, init_scale)
            # BottleFC(hidden_dim, 16, init_scale)
        )
        self.query_idx = query_idx
        self.temperature = temperature
        
    def forward(self, *xs):
        # if self.is_attn:
        # xs shape (current_depth, batch_size, hidden_dim, height, width)
        squeezed = [self.squeeze(x) for x in xs]
        squeezed = torch.stack(squeezed)  
#             current_depth, batch_size, hidden_dim, _, _ = xs.shape
#             xs2 = Rearrange('d b h w1 w2 -> (d b) h w1 w2', d = current_depth, b = batch_size)(xs)
#             squeezed = self.squeeze(xs2)
#             squeezed = Rearrange('(d b) h -> d b h', d = current_depth, b = batch_size)(squeezed)

#             query_idx = max(- current_depth, self.query_idx)
#             query_idx = current_depth // 2
        # query_idx = random.randint(0, current_depth-1)
        # squeezed_mean = squeezed.mean(dim=0)
        attn = torch.einsum('d b h, b h -> d b', squeezed, squeezed[self.query_idx,:,:])  # 
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=0)

        next_x = xs[0] * attn[0, :, None, None, None]
        for i, x in enumerate(xs[1:]):
            next_x = next_x + x * attn[i, :, None, None, None]

#             next_x = torch.einsum('d b h v w, d b -> b h v w', xs, attn)
#         else:
#             next_x = xs[0]
#             for x in xs[1:]:
#                 next_x = next_x + x
# #             next_x = torch.einsum('d b h v w -> b h v w', xs)
        return next_x


class AttnLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        drop_rate: float = 0.,
        temperature: float = 1.,
        init_scale: float = 1.,
        query_idx: int = 1,
    ):
        super().__init__()
        self.attn = Attn(hidden_dim, temperature, init_scale, query_idx)
        self.unit = Block(hidden_dim, kernel_size, drop_rate)

    def forward(self, *xs):
        x = self.attn(*xs)
        x= self.unit(x)
        return xs + (x,)


@dataclass
class DeepAttnCfg(BaseCfg):
    hidden_dim: int = 128
    kernel_size: int = 5
    patch_size: int = 2
    num_classes: int = 10
    drop_rate: float=0.
    
    is_attn: bool = True
    query_idx_exp: float = 1.
    query_idx_denominator: int = 1
    query_idx_shift: int = 0
    temperature: float = 1.
    init_scale: float = 1.



class DeepAttn(BaseModule):
    def __init__(self, cfg:DeepAttnCfg):
        super().__init__(cfg)
        
        query_idxs = [
            max(0, int(current_depth ** cfg.query_idx_exp // cfg.query_idx_denominator) - cfg.query_idx_shift)
            for current_depth in range(cfg.num_layers + 1)
        ]
        self.layers = nn.ModuleList([
            AttnLayer(cfg.hidden_dim, cfg.kernel_size, cfg.drop_rate,
                      cfg.temperature, cfg.init_scale, query_idx)
            for query_idx in query_idxs[:-1]
        ])
        
        self.final_attn = Attn(cfg.hidden_dim, is_attn=cfg.is_attn, query_idx=query_idxs[-1], temperature=cfg.temperature, init_scale=cfg.init_scale)
        
        self.embed = nn.Sequential(
            nn.Conv2d(3, cfg.hidden_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size),
            nn.GELU(),
            nn.BatchNorm2d(cfg.hidden_dim),  # , eps=7e-5
        )
        
        self.digup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(cfg.hidden_dim, cfg.num_classes)
        )

        self.cfg = cfg

    def forward(self, x):
        x = self.embed(x)
        xs = (x,)
        for layer in self.layers:
            xs = layer(*xs)
        x = self.final_attn(*xs)

#         xs = x.unsqueeze(0)
#         for layer in self.layers:
#             xs = layer(xs)
#         x= self.final_attn(xs)
        x= self.digup(x)
        return x

    def _step(self, batch, mode="train"):  # or "val"
        input, target = batch
        logits = self.forward(input)
        loss = F.cross_entropy(logits, target)
        self.log(mode + "_loss", loss, prog_bar=True)
        accuracy = (logits.argmax(dim=1) == target).float().mean()
        self.log(mode + "_accuracy", accuracy, prog_bar=True)
        return loss

