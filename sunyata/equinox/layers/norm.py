from typing import Union, Tuple

import jax
import jax.numpy as jnp

import equinox as eqx
from equinox import static_field
from torch import layer_norm

class LayerNorm(eqx.Module):
    scale: jnp.ndarray
    bias: jnp.ndarray = None
    
    normalized_shape: Union[int, Tuple[int, ...]] = eqx.static_field()  # started from the last dimension
    use_bias: bool = eqx.static_field()
    
    def __init__(self, normalized_shape, scale=1., elementwise_affine=True, use_bias=True):
        super().__init__()
        if elementwise_affine:
            self.scale = jnp.ones(normalized_shape) * scale
            if use_bias:
                self.bias = jnp.zeros(normalized_shape)
        else:
            self.scale = jnp.array(scale)
            if use_bias:
                self.bias = jnp.zeros(1)
            
        self.normalized_shape = normalized_shape
        self.use_bias = use_bias
                
    def __call__(self, x):
        axes = [-(i + 1) for i in range(len(self.normalized_shape))]
        x_norm = jax.nn.normalize(x, axes)
        x_norm = x_norm * self.scale
        if self.use_bias:
            x_norm = x_norm + self.bias
        return x_norm
        

if __name__ == '__main__':
    inputs = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 4))

    layer_norm = LayerNorm(normalized_shape=(4,), scale=1e-5)

    assert layer_norm.scale.shape == (4,)