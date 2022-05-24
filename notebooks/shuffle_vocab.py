# %%
import random, json
# %%
vocab_file = '.data/wikitext/wikitext-2-vocab-1000/vocab_std.json'
start_id = 4
vocab_size = 1000
# %%
with open(vocab_file, encoding='utf-8') as f:
    vocab = json.load(f)
    
# %%
ids = list(range(start_id, vocab_size))
random.shuffle(ids)

# %%
shuffled_vocab = [(k, ids[i-start_id]) for i, (k, v) in enumerate(vocab.items()) if v >= start_id]
shuffled_vocab = dict(shuffled_vocab)

# %%
# %%
special_tokens = dict([(k, v) for k, v in vocab.items() if v < start_id])
special_tokens.update(shuffled_vocab)
shuffled_vocab = special_tokens
# shuffled_vocab.update(special_tokens)
# %%
# vocab_items = list(vocab.items())

# random.shuffle(vocab_items)

# vocab = dict(vocab_items)

# %%
new_vocab_file = '.data/wikitext/wikitext-2-vocab-1000/vocab.json'


# %%
with open(new_vocab_file, mode='w', encoding='utf-8') as f:
    json.dump(shuffled_vocab, f, ensure_ascii=False)
# %%
