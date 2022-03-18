"""
NAME
----
    lm - Language Models

DESCRIPTION
-----------
    Data preprocessing and postprocessing needed by language models:
        For Causal Language Model (CLM), two cases. For a token sequence with length T:
        1. take all except the last token as the input,
           while, take all except the first token as the target.
        2. prefix an unknown token (uniform prob.) to the sequence. 
           Then for the new sequence, follow the steps of the above case.

CLASSES
-------
    PrePostProcessLM

"""

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from sunyata.utils import one_hot_encode

@dataclass
class PrePostProcessLM:
    """
    Attributes
    ----------
    language_model: str
        supported language models: causal language model, two cases
    dataset: str
        supported dataset: openwebtext2
    dim_categorical_probabilities: int
        dimension of categorical probability
    """
    language_model: str = "clm1"
    dataset: str = "openwebtext2"
    dim_categorical_probabilities: int = 50257

    def preprocess(self, batch: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """get input and target of the model from one batch data"""
        if self.dataset == "openwebtext2":
            batch = jnp.array(batch['text'])  # from numpy.ndarray to jax.numpy.ndarray
            input = one_hot_encode(batch, self.dim_categorical_probabilities)
            if self.language_model == "clm1":
                target = input[:, 1:, :]
                return input, target
            elif self.language_model == "clm2":
                prefix = jnp.ones_like(input[:, :1, :]) / self.dim_categorical_probabilities
                input = jnp.concatenate([prefix, input], axis=-2)
                new_input = input[:, :-1, :]
                new_target = input[:, 1:, :]
                return new_input, new_target
        else:
            raise Exception("Still does not implement!")

    def postprocess(self, output: jnp.ndarray) -> jnp.ndarray:
        """get predicted from output of the model"""
        if self.dataset == "openwebtext2":
            if self.language_model == "clm1":
                predicted = output[:, :-1, :]
                return predicted
            elif self.language_model == "clm2":
                predicted = output
                return predicted
        else:
            raise Exception("Still does not implement!")