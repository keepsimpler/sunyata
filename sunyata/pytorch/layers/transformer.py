import torch
import torch.nn as nn
from sunyata.pytorch.layers.attention import Attention
from sunyata.pytorch.bayes.core import DeepBayesInferCfg


class TransformerLayer(nn.Module):
    def __init__(self, cfg:DeepBayesInferCfg):
        super().__init__()
        self.attention = Attention(cfg.hidden_dim, cfg.num_heads, cfg.attn_scale, cfg.attn_dropout, cfg.is_to_qkv, 
            cfg.is_to_qkv_bias, cfg.is_to_out, cfg.is_to_out_bias, cfg.is_mask, cfg.is_softmax) if cfg.is_attn else nn.Identity()

        self.feed_forward = FeedForward(cfg.hidden_dim, cfg.expanded_dim, cfg.ff_dropout, cfg.ff_act_nn, 
            cfg.is_ff_bias, cfg.is_nonlinear) if cfg.is_ff else nn.Identity()
        
        if cfg.normalized_ndim == 1:
            normalized_shape = cfg.hidden_dim
        elif cfg.normalized_ndim == 2:
            normalized_shape = cfg.hidden_dim  #  (cfg.seq_len, cfg.hidden_dim)
        else:
            raise Exception('normalized_ndim must be 1 or 2')

        self.pre_layernorm = nn.LayerNorm(normalized_shape) if cfg.is_pre_layernorm else nn.Identity()
        self.inner_layernorm = nn.LayerNorm(normalized_shape) if cfg.is_inner_layernorm else nn.Identity()
        self.post_layernorm = nn.LayerNorm(normalized_shape) if cfg.is_post_layernorm else nn.Identity()

        self.is_attn_shortcut, self.is_ff_shortcut = cfg.is_attn_shortcut, cfg.is_ff_shortcut

    def forward(self, x):
        _x = self.pre_layernorm(x)

        if self.is_attn_shortcut:
            x = x + self.attention(_x)
        else:
            x = self.attention(_x)

        _x = self.inner_layernorm(x)

        if self.is_ff_shortcut:
            x = x + self.feed_forward(_x)
        else:
            x = self.feed_forward(_x)

        x = self.post_layernorm(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expanded_dim, dropout=0., act_nn=nn.GELU(), bias=True, is_nonlinear=True):
        super().__init__()
        if is_nonlinear:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, expanded_dim, bias=bias),
                act_nn,
                nn.Dropout(dropout),
                nn.Linear(expanded_dim, hidden_dim, bias=bias),
                nn.Dropout(dropout)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)
