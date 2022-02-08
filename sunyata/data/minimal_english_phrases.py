import jax.numpy as jnp
from sunyata.utils import onehot

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
    return onehot(index, len(vocab))

def tensor_to_phrase(phrase, vocab):
    index = jnp.argmax(phrase, axis=1)
    return [list(vocab)[i] for i in index]

vocab = get_vocab(phrases)
data = jnp.stack([phrase_to_tensor(phrase, vocab) for phrase in phrases])