from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DeepBayesInferCfg:
    hidden_dim: int = None

    # transformer
    is_attn=True
    is_ff=True

    # attention
    num_heads: int = None
    attn_scale: float = None
    attn_dropout: float = 0.
    is_to_qkv_bias=False
    is_to_out=True
    is_to_out_bias=False
    is_mask=True
    is_softmax=True

    # feed forward
    expanded_dim: int = None
    ff_dropout=0.
    ff_act_nn=nn.ReLU()
    ff_bias=True
    is_nonlinear=True

    # layernorm
    is_pre_layernorm=False
    is_inner_layernorm=False
    is_post_layernorm=False
    normalized_ndim = 1

    # shortcut
    is_attn_shortcut=True
    is_ff_shortcut=True

    # bayes
    is_layernorm_before_digup: bool = True
    is_sharing_weight: bool = False
    is_prior_as_params: bool = False

    # training
    batch_size: int = None
    learning_rate: float = None



def log_bayesian_iteration(log_prior: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    log_total_evidence = torch.logsumexp(log_prior + logits, dim=-1, keepdim=True)
    log_posterior = log_prior + logits - log_total_evidence
    return log_posterior


