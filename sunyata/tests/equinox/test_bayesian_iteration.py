import jax.numpy as jnp

from sunyata.equinox.archs import ComputeAbsoluteValues
from sunyata.equinox.layers import BayesianIteration

def test_bayesian_iteration():
    dim_categorical_probabilities = 6

    priors = jnp.ones(dim_categorical_probabilities) / dim_categorical_probabilities

    candidate_evidences = jnp.arange(-dim_categorical_probabilities//2, dim_categorical_probabilities//2)

    map_to_non_negative_by_compute_absolute_values = ComputeAbsoluteValues()
    evidences = map_to_non_negative_by_compute_absolute_values(candidate_evidences)

    bayesian_iteration = BayesianIteration()
    posteriors = bayesian_iteration(priors, evidences)

    assert jnp.allclose(jnp.sum(posteriors), 1.)
