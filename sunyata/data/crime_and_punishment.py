# %%
import jax.numpy as jnp
from jax import random
import numpy as np
from sentencepiece import SentencePieceProcessor

from sunyata.utils import smoothing_func

# %%
def get_crime_and_punishment(file_path):
    with open(file_path + '/crime-and-punishment-2554.txt') as f:
        text = f.read()
    # The file read above includes metadata and licensing information.
    # For training our language model, we will only use the actual novel text.
    start = text.find('CRIME AND PUNISHMENT')  # skip header
    start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip header
    start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip translator preface
    end = text.rfind('End of Project')  # skip extra text at the end
    text = text[start:end].strip()

    tokenizer = SentencePieceProcessor()
    tokenizer.load(file_path + '/cp.320.model')
    ids = tokenizer.EncodeAsIds(text)  # all the token ids
    ids = np.asarray(ids, dtype=np.int32)

    return text, tokenizer, ids
# %%
def data_loader(key, input_ids, seq_len, batch_size, vocab_size, 
                inputs_smoothing_ratio, targets_smoothing_ratio, 
                random_inputs: float=0., shuffle=False):
    steps_per_epoch = len(input_ids) // (seq_len * batch_size)
    vocab_mean = 1 / vocab_size
    input_ids = input_ids[: steps_per_epoch * batch_size * seq_len]
    input_ids = input_ids.reshape(steps_per_epoch * batch_size, seq_len)
    if shuffle:
        batch_idx = random.permutation(key, steps_per_epoch * batch_size)
    else:
        batch_idx = jnp.arange(steps_per_epoch * batch_size)
        
    batch_idx = batch_idx.reshape(steps_per_epoch, batch_size)
    
    for idx in batch_idx:
        batch = input_ids[idx]
        # batch shape (batch_size, seq_len + 1)
        if random_inputs > 0:
            inputs = smoothing_func(batch, vocab_size, inputs_smoothing_ratio)
            mask_matrix = np.random.uniform(0., 1., (batch_size, seq_len)) < random_inputs
            inputs[mask_matrix] = (1 / vocab_size)
            targets = smoothing_func(batch, vocab_size, targets_smoothing_ratio)
            targets[np.bool_(1 - mask_matrix)] = 0.
#             inputs = np.random.uniform(-vocab_mean, vocab_mean, (batch_size, seq_len-1, vocab_size))
#             inputs = random.uniform(key, (batch_size, seq_len-1, vocab_size), minval = -vocab_mean, maxval=vocab_mean)
#             inputs = jax.nn.softmax(inputs, axis=-1)
        else:
            inputs = smoothing_func(batch[:, :-1], vocab_size, inputs_smoothing_ratio)
            pre = jnp.ones((batch_size, 1, vocab_size)) / vocab_size
            inputs = jnp.concatenate([pre, inputs], axis=1)
            targets = smoothing_func(batch, vocab_size, targets_smoothing_ratio)
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)
        
        yield (batch, inputs, targets)

# %%
if __name__ == '__main__':
    key = random.PRNGKey(1)
    text, tokenizer, ids = get_crime_and_punishment('resources/')
    train_loader = data_loader(key, ids, seq_len=6, batch_size=2,
                           vocab_size=tokenizer.vocab_size(), 
                           inputs_smoothing_ratio=0., 
                           targets_smoothing_ratio=0., random_inputs=0.)
# %%
