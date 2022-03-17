import jax
import jax.numpy as jnp
from jax import random

from sunyata.equinox.archs import (
    ComputeAbsoluteValues, ComputeExponentialValues,
    ComputeReluAndSquaredValues, ComputeSquaredValues,
    MapBetweenCategoricalProbabilitiesAndHiddenFeatures)


def test_map_between_categorical_probabilities_and_hidden_features():
    key = random.PRNGKey(1)
    dim_categorical_probabilities, dim_hidden_features = 16, 8
    embed_and_digup = MapBetweenCategoricalProbabilitiesAndHiddenFeatures(
                                                        key, dim_categorical_probabilities, dim_hidden_features,
                                                        weight_init_func=jax.nn.initializers.glorot_normal())


    batch_size, sequence_len = 2, 32
    categorical_probabilities = random.normal(key, (batch_size, sequence_len, dim_categorical_probabilities))
    hidden_features = embed_and_digup.embed(categorical_probabilities)
    assert hidden_features.shape == (batch_size, sequence_len, dim_hidden_features)

    candidate_evidences = embed_and_digup.digup(hidden_features)
    assert candidate_evidences.shape == (batch_size, sequence_len, dim_categorical_probabilities)

    compute_absolute_values = ComputeAbsoluteValues()
    evidences = compute_absolute_values(candidate_evidences)
    assert jnp.all(evidences) >= 0

    compute_squared_values = ComputeSquaredValues()
    evidences = compute_squared_values(candidate_evidences)
    assert jnp.all(evidences) >= 0

    compute_exponential_values = ComputeExponentialValues()
    evidences = compute_exponential_values(candidate_evidences)
    assert jnp.all(evidences) >= 0

    compute_relu_and_squared_values = ComputeReluAndSquaredValues()
    evidences = compute_relu_and_squared_values(candidate_evidences)
    assert jnp.all(evidences) >= 0
