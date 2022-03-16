# %%
from typing import List
import jax
import jax.numpy as jnp
import equinox as eqx
from sunyata.equinox.archs import MapBetweenCategoricalProbabilitiesAndHiddenFeatures, MapValuesToNonNegative
from sunyata.equinox.layers import BayesianIteration

# %%
class HiddenBayesianNet(eqx.Module):
    """
    An architecture worked both in the categorical probability space and the hidden feature space, that :
        initially, initialize uniform priors in the space of V-dimensional categorical probabilities;
        then:
            first, embeds data samples in the space of V-dimensional categorical probabilities to a space of D-dimensional hidden feature vectors;
            then, transform in the space of D-dimensional hidden feature vectors layer by layer;
            after each layer: 
                1. dig up from the space of D-dimensional hidden feature vectors back to the space of V-dimensional categorical probabilities.
                2. uses digged data as evidences, and the previous priors, executes one iteration of Bayes' theorem, gets posteriors.
                3. uses posteriors as new priors.
        finally, output the last priors.

    Attributes
    ----------
    embed_and_digup: MapBetweenCategoricalProbabilitiesAndHiddenFeatures

    map_values_to_non_negative: MapValuesToNonNegative

    bayesian_iteration: BayesianIteration

    layers: List[eqx.Module]

    """

    embed_and_digup: MapBetweenCategoricalProbabilitiesAndHiddenFeatures
    map_values_to_non_negative: MapValuesToNonNegative

    bayesian_iteration: BayesianIteration

    layers: List[eqx.Module]

    def __call__(self, categorical_probabilities: jnp.ndarray):
        dim_categorical_probabilities = categorical_probabilities.shape[-1]

        hidden_features = self.embed_and_digup.embed(categorical_probabilities)

        priors = jnp.ones_like(categorical_probabilities) / dim_categorical_probabilities

        for layer in self.layers:
            hidden_features = layer(hidden_features)
            candidate_evidences = self.embed_and_digup.digup(hidden_features)
            evidences = self.map_values_to_non_negative(candidate_evidences)
            posteriors = self.bayesian_iteration(priors, evidences)
            priors = posteriors

        return priors