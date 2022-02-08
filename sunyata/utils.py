import numpy as np

def smoothing_func(batch: np.ndarray, vocab_size: int, smoothing_ratio: float):
    x = np.ones(batch.shape + (vocab_size,)) * smoothing_ratio / vocab_size
    d1 = np.arange(batch.shape[0])  # batch_size
    d2 = np.arange(batch.shape[1])  # seq_len
    d3 = batch.reshape(-1)
    x[np.repeat(d1, len(d2)), np.tile(d2, len(d1)), d3] = 1 - smoothing_ratio + smoothing_ratio / vocab_size
#     x = jnp.ones((N,)) * smoothing_ratio / N
#     x = x.at[K].set(1 - smoothing_ratio + smoothing_ratio / N)
    return x