import jax
import jax.numpy as jnp
from jax import lax, random
import equinox as eqx

from typing import Callable


class MapBetweenCategoricalProbabilitiesAndHiddenFeatures(eqx.Module):
    """
    Map from V-dimensional categorical probabilities to D-dimensional hidden features, and vice versa.

    Attributes
    ----------
    weight : jnp.ndarray with shape [dim_categorical_probabilities, dim_hidden_features]
        the transition matrix used both at embedding and at digging up.

    Methods
    -------
    embed(categorial_probabilities: jnp.ndarray with shape [batch_size, sequence_len, dim_categorical_probabilities]):
        embedding categorical probabilities to hidden features.

    digup(hidden_features: jnp.ndarray with shape [batch_size, sequence_len, dim_hidden_features]):
        dig up categorical probabilities from hidden features 

    """

    weight: jnp.ndarray

    def __init__(self, key: random.PRNGKey, dim_categorical_probabilities: int, dim_hidden_features: int, weight_init_func: Callable):
        super().__init__()
        self.weight = weight_init_func(key, (dim_categorical_probabilities, dim_hidden_features))
        
    def embed(self, categorial_probabilities: jnp.ndarray):
        hidden_features = jnp.einsum("b s v, v d -> b s d", categorial_probabilities, self.weight)
        return hidden_features
        
    def digup(self, hidden_features: jnp.ndarray):
        evidences = jnp.einsum("v d, b s d -> b s v", self.weight, hidden_features)
        return evidences


# %%
class MapValuesToNonNegative:  # (eqx.Module)
    """Map values to non negative, in order to work as evidences."""
    def __call__(self, x: jnp.ndarray):
        pass

class ComputeAbsoluteValues(MapValuesToNonNegative):
    """Map values to non negative by computing absolute values"""
    def __call__(self, x: jnp.ndarray):
        return jnp.abs(x)

class ComputeSquaredValues(MapValuesToNonNegative):
    def __call__(self, x: jnp.ndarray):
        return x ** 2

class ComputeExponentialValues(MapValuesToNonNegative):
    def __call__(self, x: jnp.ndarray):
        return jnp.exp(x)

class ComputeReluAndSquaredValues(MapValuesToNonNegative):
    def __call__(self, x: jnp.ndarray):
        return jax.nn.relu(x) ** 2

