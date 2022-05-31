from dataclasses import dataclass
import torch
import torch.nn as nn
from sunyata.pytorch.layer.attention import Attention


@dataclass
class TransformerCfg:
    hidden_dim: int = None

     # attention
    is_attn: bool = True
    num_heads: int = None  # 16
    attn_scale: float = None
    attn_dropout: float = 0.
    is_mask: bool = True
    is_softmax: bool = True

    # feed forward
    is_ff: bool = True
    expanded_dim: int = None  # 2048
    ff_dropout: float = 0.
    ff_act_nn: nn.Module = nn.GELU()

    # layernorm
    is_attn_layernorm: bool = True
    is_ff_layernorm: bool = True

    # shortcut
    # is_attn_shortcut: bool = True
    # is_ff_shortcut: bool = True


class TransformerLayer(nn.Module):
    def __init__(self, cfg:TransformerCfg):
        super().__init__()
        self.attention = Attention(cfg.hidden_dim, cfg.num_heads, cfg.attn_scale, cfg.attn_dropout,
                                    cfg.is_mask, cfg.is_softmax) if cfg.is_attn else nn.Identity()

        self.feed_forward = FeedForward(cfg.hidden_dim, cfg.expanded_dim, cfg.ff_act_nn, 
                                        cfg.ff_dropout) if cfg.is_ff else nn.Identity()
        

        self.attn_layernorm = nn.LayerNorm(cfg.hidden_dim) if cfg.is_attn_layernorm else nn.Identity()
        self.ff_layernorm = nn.LayerNorm(cfg.hidden_dim) if cfg.is_ff_layernorm else nn.Identity()

    def forward(self, x):
        x = x + self.feed_forward(self.ff_layernorm(x))
        x = x + self.attention(self.attn_layernorm(x))
        return x
        

class TransformerLayerNoShortcut(TransformerLayer):
    def __init__(self, cfg:TransformerCfg):
        super().__init__(cfg)

    def forward(self, x):
        x1 = self.attn_layernorm(self.attention(x))
        x2 = self.ff_layernorm(self.feed_forward(x1))
        return x1, x2


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expanded_dim, act_nn=nn.GELU(), dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            act_nn,
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
