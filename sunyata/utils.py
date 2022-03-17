import glob
from typing import List
import jax.numpy as jnp
from jax import lax

import numpy as np

def get_all_files_with_specific_filetypes_in_a_directory(directory: str, filetypes: List[str]=["*"]):
    files_with_specific_filetypes = []
    for filetype in filetypes:
        files_with_specific_filetypes.extend(glob.glob(directory + f"*.{filetype}"))
        
    return files_with_specific_filetypes

def smoothing_func(batch: np.ndarray, vocab_size: int, smoothing_ratio: float):
    x = np.ones(batch.shape + (vocab_size,)) * smoothing_ratio / vocab_size
    d1 = np.arange(batch.shape[0])  # batch_size
    d2 = np.arange(batch.shape[1])  # seq_len
    d3 = batch.reshape(-1)
    x[np.repeat(d1, len(d2)), np.tile(d2, len(d1)), d3] = 1 - smoothing_ratio + smoothing_ratio / vocab_size
#     x = jnp.ones((N,)) * smoothing_ratio / N
#     x = x.at[K].set(1 - smoothing_ratio + smoothing_ratio / N)
    return x


# copy from flax.training.common_utils.onehot
def one_hot(seq, vocab_size, on_value=1.0, off_value=0.0):
    x = (seq[..., None] == jnp.arange(vocab_size).reshape((1,) * seq.ndim + (-1,)))
    x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return x.astype(jnp.float32)