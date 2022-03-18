from sunyata.equinox.train import PrePostProcessLM
import jax.numpy as jnp

def test_pre_post_process_lm():
    pre_post_process_lm = PrePostProcessLM(language_model="clm1", dataset="openwebtext2", dim_categorical_probabilities=50257)

    batch_size, sequence_len, dim_categorical_probabilities = 2, 16, 50257

    batch = {
        "text": jnp.zeros((batch_size, sequence_len))
    }

    input, target = pre_post_process_lm.preprocess(batch)
    assert input.shape == (batch_size, sequence_len, dim_categorical_probabilities)
    assert target.shape == (batch_size, sequence_len - 1, dim_categorical_probabilities)

    output = input
    predicted = pre_post_process_lm.postprocess(output)
    assert predicted.shape == (batch_size, sequence_len - 1, dim_categorical_probabilities)


    pre_post_process_lm = PrePostProcessLM(language_model="clm2", dataset="openwebtext2", dim_categorical_probabilities=50257)

    input, target = pre_post_process_lm.preprocess(batch)
    assert input.shape == (batch_size, sequence_len, dim_categorical_probabilities)
    assert target.shape == (batch_size, sequence_len, dim_categorical_probabilities)

    output = input
    predicted = pre_post_process_lm.postprocess(output)
    assert predicted.shape == (batch_size, sequence_len, dim_categorical_probabilities)