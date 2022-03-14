# %%
import jax
import jax.numpy as jnp
import equinox as eqx
from numpy import ndarray

# %%
class MapValuesToNonNegative(eqx.Module):
    def __call__(self, x: jnp.ndarray):
        pass

# %%
class ByComputeAbsoluteValues(MapValuesToNonNegative):
    """Map values to non negative by computing absolute values"""
    def __call__(self, x: jnp.ndarray):
        return jnp.abs(x)

# %%
class ByComputeSquaredValues(MapValuesToNonNegative):
    def __call__(self, x: jnp.ndarray):
        return x ** 2

# %%
class ByComputeExponentialValues(MapValuesToNonNegative):
    def __call__(self, x: jnp.ndarray):
        return jnp.exp(x)

# %%
class ByComputeReluAndSquaredValues(MapValuesToNonNegative):
    def __call__(self, x: jnp.ndarray):
        return jax.nn.relu(x) ** 2


# %%
class BayesianIteration(eqx.Module):
    """One iteration of Bayes' theorem, get posteriors from priors and evidences"""
    def __call__(self, priors: jnp.ndarray, evidences: jnp.ndarray) -> jnp.ndarray:
        total_evidence = jnp.sum(priors * evidences, axis=-1, keepdims=True)
        posteriors = (priors * evidences) / total_evidence
        
        return posteriors

# %%
if __name__ == '__main__':
    priors = jnp.ones(6) / 6
    x = jnp.array([-2, -1, 0, 1, 2, 3])

    map_to_non_negative_by_compute_absolute_values = ByComputeAbsoluteValues()
    evidences = map_to_non_negative_by_compute_absolute_values(x)
    print(evidences)

    bayesian_iteration = BayesianIteration()
    posteriors = bayesian_iteration(priors, evidences)
    print(posteriors)
# %%
