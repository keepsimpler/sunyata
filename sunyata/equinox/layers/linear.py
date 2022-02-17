from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax import lax, random

import equinox as eqx
from equinox import static_field

class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    
    use_bias: bool = static_field()
    
    def __init__(self, key: random.PRNGKey, in_features: int, out_features: int,  
                 weight_init_func: Callable=jax.nn.initializers.xavier_normal(),
                 use_bias=True, bias_init_func: Callable=jax.nn.initializers.zeros):
        super().__init__()
        self.weight = weight_init_func(key, (in_features, out_features))
        if use_bias:
            self.bias = bias_init_func(key, out_features)
        else:
            self.bias = None
        
        self.use_bias = use_bias

    def __call__(self, x):
        x = lax.dot_general(x, self.weight,
                            (((x.ndim - 1,), (2 - 2,)), ((), ())),)
        if self.use_bias:
            x = x + self.bias
            
        return x
        

class LinearWithMask(Linear):
    def __call__(self, x):
        weight = jnp.triu(self.weight)
        x = lax.dot_general(x, weight,
                            (((x.ndim - 1,), (2 - 2,)), ((), ())),)
        if self.use_bias:
            x = x + self.bias
            
        return x


class LinearMixer(eqx.Module):
    linear_vocab: Linear
    linear_seq: LinearWithMask
    
    def __call__(self, x):
        x_vocab = self.linear_vocab(x)
        x_vocab_transpose = jnp.transpose(x_vocab, (0, 2, 1))

        x_vocab_transpose_seq = self.linear_seq(x_vocab_transpose)
        x_vocab_transpose_seq_transpose = jnp.transpose(x_vocab_transpose_seq, (0, 2, 1))
        
        return x_vocab_transpose_seq_transpose


class Embedding(eqx.Module):
    weight: jnp.ndarray
    
    def __init__(self, vocab_size: int, embed_size: int, init_func: Callable, key: random.PRNGKey):
        super().__init__()
        self.weight = init_func(key, (vocab_size, embed_size))
        
    def __call__(self, x):
        # x_embed = jnp.take(self.weight, x, axis=0)
        x_embed = self.weight[x]        
        return x_embed


if __name__ == '__main__':
    key = random.PRNGKey(1)
    linear_vocab_key, linear_seq_key = random.split(key, num=2)
    vocab_size = 15
    seq_len = 6
    linear_vocab = Linear(key=linear_vocab_key, in_features=vocab_size, out_features=vocab_size,
                      use_bias=False)
    linear_seq = LinearWithMask(key=linear_seq_key, in_features=seq_len, out_features=seq_len,
                        use_bias=False)
    linear_mixer = LinearMixer(linear_vocab, linear_seq)

    data = jnp.ones((2, seq_len, vocab_size))
    output = linear_mixer(data)
    print(output.shape)