# %%
from datasets import load_dataset, load_from_disk
from tokenizers import ByteLevelBPETokenizer
path = "resources/"
language = "is"  # the Icelandic language

data_dir = path + f"oscar-{language}/"
# %%
dataset = load_dataset("wikitext", "wikitext-103-v1")
# %%
raw_dataset = load_dataset("oscar", f"unshuffled_deduplicated_{language}")
# %%
grouped_datasets = load_from_disk(data_dir + "error")
# %%
len(raw_dataset["train"]['text'])
# %%
# %%
text_lens = [len(text) for text in raw_dataset['train']['text']]
# %%
max(text_lens), min(text_lens)
# %%
sorted_text_lens = sorted(text_lens)
# %%
sorted_text_lens[-10000:]
# %%
tokenizer = ByteLevelBPETokenizer()
# %%
def batch_iterator(dataset, batch_size=1000):
  for i in range(0, len(dataset["train"]), batch_size):
    yield dataset["train"][i: i + batch_size]["text"]
# %%
vocab_size = 20000
tokenizer.train_from_iterator(batch_iterator(dataset), vocab_size=vocab_size, min_frequency=3)
# %%
vocab = tokenizer.get_vocab()
# %%
outputs = tokenizer.encode(dataset['train'][3]['text'])
# %%
outputs = tokenizer.encode("Hello, y'all! How are you .")
# %%
outputs.tokens, outputs.ids
# %%
vocab_ = sorted(vocab.items(), key=lambda item: item[1])
# %%
