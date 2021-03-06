import jax
import jax.numpy as jnp
from jax import random
from sunyata.equinox.archs import (
    ComputeExponentialValues, HiddenBayesianNet,
    MapBetweenCategoricalProbabilitiesAndHiddenFeatures)
from sunyata.equinox.layers import BayesianIteration
from sunyata.equinox.layers import Linear

def test_hidden_bayesian_net():

    key = random.PRNGKey(1)
    dim_categorical_probabilities, dim_hidden_features = 16, 8
    embed_and_digup = MapBetweenCategoricalProbabilitiesAndHiddenFeatures(
                                                        key, dim_categorical_probabilities, dim_hidden_features,
                                                        weight_init_func=jax.nn.initializers.glorot_normal())


    layers = [Linear(key, in_features=dim_hidden_features, out_features=dim_hidden_features) for _ in range(2)]

    hidden_bayesian = HiddenBayesianNet(embed_and_digup, layers)

    batch_size, sequence_len = 2, 32
    categorical_probabilities = random.normal(key, (batch_size, sequence_len, dim_categorical_probabilities))

    posteriors = hidden_bayesian(categorical_probabilities)
    assert posteriors.shape == (batch_size, sequence_len, dim_categorical_probabilities)

    assert jnp.allclose(jnp.sum(posteriors, axis=-1), 1.)

