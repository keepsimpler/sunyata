# %%
from datasets import load_dataset, load_from_disk
from tokenizers import ByteLevelBPETokenizer
path = ".data/"

data_dir = path + f"wikitext-103-v1/"
# %%
dataset = load_dataset("wikitext", "wikitext-103-v1")
# %%
text_lens = [len(text) for text in dataset['train']['text']]
# %%
max(text_lens), min(text_lens)

# %%
tokenizer = ByteLevelBPETokenizer()
# %%
def batch_iterator(dataset, batch_size=1000):
  for i in range(0, len(dataset["train"]), batch_size):
    yield dataset["train"][i: i + batch_size]["text"]
# %%
vocab_size = 20000
tokenizer.train_from_iterator(batch_iterator(dataset), vocab_size=vocab_size, min_frequency=2, special_tokens=["<unk>","\n"])
# %%
# %%
outputs = tokenizer.encode(dataset['train'][3]['text'])
# %%
outputs = tokenizer.encode("Hello, y'all! How are you.")
# %%
outputs.tokens, outputs.ids
# %%
vocab = tokenizer.get_vocab()
vocab_ = sorted(vocab.items(), key=lambda item: item[1])
# %%
tokenizer.save(f"{path}wikitext-103-v{vocab_size}.json")
# %%
encoded_dataset = [tokenizer.encode(line.strip(' ')) for line in dataset['train']['text'] if len(line) > 0]
# %%
encoded_dataset