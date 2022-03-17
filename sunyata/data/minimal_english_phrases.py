import jax.numpy as jnp
from sunyata.utils import one_hot_encode

# a set of six word phrases
phrases = [
    'the quick brown fox jumped over ',
    'the cat sat on the mat ',
    'four legs good two legs bad '
]

def get_vocab(phrases):
    vocab = {}
    for word in "".join(phrases).split():
        vocab[word] = None
    for i, key in enumerate(vocab):
        vocab[key] = i
    return vocab

def phrase_to_tensor(phrase, vocab):
    tokens = [vocab[word] for word in phrase.split()]
    index = jnp.array(tokens)
    return one_hot_encode(index, len(vocab))

def tensor_to_phrase(phrase, vocab):
    index = jnp.argmax(phrase, axis=1)
    return [list(vocab)[i] for i in index]

def get_minimal_english_phrases():
    vocab = get_vocab(phrases)
    data = jnp.stack([phrase_to_tensor(phrase, vocab) for phrase in phrases])
    batch_size = 3
    vocab_size = len(vocab)
    seq_len = 6
    pre = jnp.ones((batch_size, 1, vocab_size)) / vocab_size
    inputs = jnp.concatenate([pre, data], axis=1)
    inputs = inputs[:, :-1, :]
    return inputs, data, vocab, batch_size, vocab_size, seq_len

if __name__ == '__main__':
    inputs, data, vocab, batch_size, vocab_size, seq_len = get_minimal_english_phrases()
    # print(data, vocab, vocab_size, seq_len)