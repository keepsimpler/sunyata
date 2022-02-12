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
    
    def __init__(self, in_features: int, out_features: int, key: random.PRNGKey, 
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