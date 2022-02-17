from typing import Callable, List
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx

from sunyata.equinox.layers import Linear, LinearWithMask, LinearMixer, BayesianLayer

class ReplicatorLayer(eqx.Module):
    # linear_vocab: Linear
    # linear_seq: LinearWithMask
    linear_mixer: LinearMixer
    bayesian_layer: BayesianLayer
        
    def __init__(self, key: random.PRNGKey, vocab_size: int, seq_len: int, positive_method: str='none',
                 vocab_weight_init_func: Callable=jax.nn.initializers.xavier_normal(),
                 seq_weight_init_func: Callable=jax.nn.initializers.xavier_normal(),
                 use_bias=True, bias_init_func: Callable=jax.nn.initializers.zeros):
        super().__init__()
        linear_vocab_key, linear_seq_key = random.split(key, num=2)
        linear_vocab = Linear(key=linear_vocab_key, in_features=vocab_size, out_features=vocab_size,
                              weight_init_func=vocab_weight_init_func, use_bias=use_bias,
                              bias_init_func=bias_init_func)
        linear_seq = LinearWithMask(key=linear_seq_key, in_features=seq_len, out_features=seq_len,
                              weight_init_func=seq_weight_init_func, use_bias=use_bias,
                              bias_init_func=bias_init_func)
        self.linear_mixer = LinearMixer(linear_vocab, linear_seq)
        self.bayesian_layer = BayesianLayer(positive_method=positive_method)
        
    def __call__(self, priors, evidence):
        new_evidence = self.linear_mixer(evidence)
        posteriors = self.bayesian_layer(priors, new_evidence)
        return posteriors, new_evidence


@dataclass
class ReplicatorCfg:
    layers_num: int
    vocab_size: int
    seq_len: int

    batch_size: int

    positive_method: str='none'  # 'abs'  'square'  'exp'
    vocab_weight_init_func: Callable=jax.nn.initializers.xavier_normal()
    seq_weight_init_func: Callable=jax.nn.initializers.xavier_normal()
    use_bias: bool=False
    bias_init_func: Callable=jax.nn.initializers.zeros

    seed: int = 1  # PRNG seed

class ReplicatorNet(eqx.Module):
    layers: List[ReplicatorLayer]
    vocab_size: int
    
    def __init__(self, cfg:ReplicatorCfg):
        super().__init__()
        key = random.PRNGKey(cfg.seed)

        layers = []
        for i in range(cfg.layers_num):
            replicator_key, key = random.split(key, num=2)
            replicator_layer = ReplicatorLayer(replicator_key, cfg.vocab_size, cfg.seq_len, cfg.positive_method,
                 cfg.vocab_weight_init_func, cfg.seq_weight_init_func, cfg.use_bias, cfg.bias_init_func)
            layers.append(replicator_layer)
        self.layers = layers

        self.vocab_size = cfg.vocab_size

    def __call__(self, evidence):
        priors = jnp.ones_like(evidence) / self.vocab_size

        for layer in self.layers:
            priors, evidence = layer(priors, evidence)

        return priors


@partial(eqx.filter_value_and_grad, has_aux=True)
def replicator_loss_fn(model, inputs, targets):
    outputs = model(inputs)
#     outputs = jnp.where(outputs==0, 2e-38, outputs)
    losses = -jnp.sum(jnp.log(outputs) * targets, axis=-1)
    loss = losses.mean()
    accuracy = jnp.mean(outputs.argmax(-1) == targets.argmax(-1))
    return loss, accuracy


if __name__ == '__main__':
    from sunyata.data.minimal_english_phrases import *
    inputs, data, vocab, batch_size, vocab_size, seq_len = get_minimal_english_phrases()

    key = random.PRNGKey(1)
    replicator_layer = ReplicatorLayer(key, vocab_size, seq_len, positive_method='exp', use_bias=False)
    init_priors = jnp.ones_like(data) / vocab_size
    posteriors, new_evidence = replicator_layer(init_priors, data)
    print(posteriors.shape, new_evidence.shape)

    cfg = ReplicatorCfg(layers_num=2, vocab_size=vocab_size, seq_len=seq_len, batch_size=2, positive_method='exp')

    replicator_net = ReplicatorNet(cfg)

    outputs = replicator_net(data)

    print(outputs.shape)