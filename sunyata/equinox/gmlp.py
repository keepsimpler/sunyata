from typing import Callable, List
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random
import optax

import equinox as eqx
from equinox import static_field

from sunyata.equinox.layers import LayerNorm, Linear, LinearWithMask, Embedding

class SGU(eqx.Module):
    "Spatial Gated Unit"
    gate_norm: LayerNorm
    gate_linear: Linear
        
    def __call__(self, x):
        x, gate = jnp.split(x, 2, axis=-1)
        gate = self.gate_norm(gate)
        gate = jnp.swapaxes(gate, -1, -2)
        gate = self.gate_linear(gate)
        gate = jnp.swapaxes(gate, -1, -2)
        
        x = x * gate
        return x


class GMLP(eqx.Module):
    sgu: SGU
    pre_norm: LayerNorm
    project_in: Linear
    project_out: Linear
    activation: Callable = static_field()
        
    def __init__(self, seq_len, embed_size, ff_size, key: random.PRNGKey, activation: Callable=jax.nn.relu):
        super().__init__()
        project_in_key, gate_linear_key, project_out_key = random.split(key, 3)
        
        self.pre_norm = LayerNorm([embed_size], scale=1., elementwise_affine=True, use_bias=True)
        self.project_in = Linear(project_in_key, embed_size, ff_size, use_bias=True)
        self.activation = activation

        gate_norm = LayerNorm([ff_size // 2], scale=1., elementwise_affine=True, use_bias=True)
        gate_linear = LinearWithMask(gate_linear_key, seq_len, seq_len, weight_init_func=jax.nn.initializers.normal(stddev=1e-4),
                             use_bias=True, bias_init_func=jax.nn.initializers.ones)        
        self.sgu = SGU(gate_norm, gate_linear)
        
        self.project_out = Linear(project_out_key, ff_size // 2, embed_size, use_bias=True)
        
    def __call__(self, x):
        shortcut = x
        x = self.pre_norm(x)
        x = self.project_in(x)
        # x = jnp.maximum(x, 0)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.project_out(x)
        
        x = x + shortcut
        return x


@dataclass
class GMLPCfg:
    vocab_size: int  # vocabulary size
    seq_len: int  # token(word) sequence length
    embed_size: int  # size of embedded feature vector
    
    ff_size: int = None  # feedforward size
    heads: int = None  # heads number
    layers_num: int = None  # number of transformer layers
    
    # layer normalization coefficient
    elementwise_affine: bool = True
    use_bias: bool = False
    
    # initialization function of parameters
    embedding_init_func: Callable = jax.nn.initializers.xavier_normal()
    project_out_init_func: Callable = jax.nn.initializers.xavier_normal()
    
    seed: int = 1  # PRNG seed    


class GMLPNet(eqx.Module):
    embedding: Embedding
    layers: List[GMLP]
    out_norm: LayerNorm
    project_out: Linear
    
    def __init__(self, cfg:GMLPCfg):
        super().__init__()
        key = random.PRNGKey(cfg.seed)
        embedding_key, project_out_key, key = random.split(key, 3)
        
        self.embedding = Embedding(cfg.vocab_size, cfg.embed_size, cfg.embedding_init_func, embedding_key)
        self.out_norm = LayerNorm((cfg.embed_size,), elementwise_affine=cfg.elementwise_affine, use_bias=cfg.use_bias)
        self.project_out = Linear(project_out_key, cfg.embed_size, cfg.vocab_size, weight_init_func=cfg.project_out_init_func)
        
        layers = []
        for i in range(cfg.layers_num):
            gmlp = GMLP(cfg.seq_len, cfg.embed_size, cfg.ff_size, key)
            layers.append(gmlp)
        self.layers = layers
        
    def __call__(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        x = self.project_out(x)
        return x


@partial(eqx.filter_value_and_grad, has_aux=True)
def gmlp_loss_fn(model, inputs, targets_onehot):
    outputs = model(inputs)
    loss = optax.softmax_cross_entropy(outputs[..., :-1,:], targets_onehot).mean()
    accuracy = jnp.mean(outputs[..., :-1,:].argmax(-1) == targets_onehot.argmax(-1))
    return loss, accuracy


if __name__ == '__main__':
    batch_size, vocab_size, seq_len = 3, 15, 6
    inputs = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), minval=0, maxval=vocab_size-1)

    cfg = GMLPCfg(vocab_size=vocab_size, seq_len=seq_len, embed_size=seq_len, ff_size=seq_len*4, heads=8, layers_num=4)

    gmlp_net = GMLPNet(cfg)
    outputs = gmlp_net(inputs)

    print(outputs.shape)