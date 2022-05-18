# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
from sunyata.pytorch.wikitext import WikiTextDataModule


# %%
hidden_dim = 64
vocab_size = 1000
seq_len = 128
batch_size = 2
# %%
wikitext2 = WikiTextDataModule(subset="2", 
                   data_dir=".data/wikitext/", 
                   batch_size=batch_size,
                   vocab_size=vocab_size,
                   seq_len=seq_len,
                   is_collate=True)
# %%
wikitext2.tokenizer.decode(wikitext2.train_data[0].tolist(), skip_special_tokens=False)
# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
# https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python

# %%
input, target = next(iter(wikitext2.train_dataloader()))
input.shape, target.shape

# %%
embed = nn.Embedding(vocab_size, hidden_dim)
embeded = embed(input)
embeded.shape
# %%

# %%
# %%
seq_len = 64
seq = torch.arange(seq_len)

# %% from one dim to two dim
kernel = 4
stride = 2
col = kernel + stride * kernel - 1
"col = ", col
# %%
idx = 0
while idx < len(seq) - col:
    print(seq[idx:idx+col])
    idx = idx + kernel


# %%
row = (seq_len + 1) // kernel - stride
"row = ", row
# %%
(row - 1) * kernel + col
(row - 1) * kernel + kernel + stride * kernel - 1
# %%
unfolded = seq.unfold(0, col, kernel)
unfolded.shape
# %%
conv2d = nn.Conv2d()
# %%
embedded = embeded.permute(0,2,1)
# %%
embedded.unfold(-1, col, kernel).shape
# %%
(127 - col) / kernel + 1
# %%


