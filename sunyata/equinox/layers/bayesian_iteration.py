import jax
import jax.numpy as jnp
import equinox as eqx

class BayesianIteration:  # (eqx.Module)
    """One iteration of Bayes' theorem, get posteriors from priors and evidences"""
    def __call__(self, priors: jnp.ndarray, evidences: jnp.ndarray) -> jnp.ndarray:
        total_evidence = jnp.sum(priors * evidences, axis=-1, keepdims=True)
        posteriors = (priors * evidences) / total_evidence
        
        return posteriors

def bayesian_iteration(priors: jnp.ndarray, evidences: jnp.ndarray) -> jnp.ndarray:
        total_evidence = jnp.sum(priors * evidences, axis=-1, keepdims=True)
        posteriors = (priors * evidences) / total_evidence
        
        return posteriors
