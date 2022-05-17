# %%
import json
from os import sep
import os
import torch
import torch.nn as nn
from transformers import BertTokenizerFast


# %%
seq = torch.arange(128)
# %% from one dim to two dim
kernel = 3
stride = 2
row = kernel + stride * kernel - 1

# %%
idx = 0
while idx < len(seq) - row:
    print(seq[idx:idx+row])
    idx = idx + kernel


# %%
from sunyata.pytorch.wikitext import WikiTextDataModule


# %%
wikitext2 = WikiTextDataModule(subset="2", 
                   data_dir=".data/wikitext/", 
                   batch_size=16,
                   vocab_size=1000,
                   seq_len=128,
                   is_collate=True)


# %% https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb
from datasets import load_dataset
# %%
dataset = load_dataset("wikitext", "wikitext-2-v1")
# %%
train_texts = dataset['train']['text']
type(train_texts), len(train_texts)
# %%
def batch_iterator(dataset: list, batch_size: int):
    for i in range(0, len(dataset), batch_size):
        yield [line.strip(' ').replace('\n', '[EOS]').replace('<unk>', '[UNK]')
                for line in dataset[i: i + batch_size]]
# %%
next(batch_iterator(train_texts, batch_size=10))
# %%
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer

# %%
tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]'))
# %%
tokenizer_normalizer = normalizers.BertNormalizer(lowercase=True)
# %%
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# %%
tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!")
# %%
special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]', '[EOS]']
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
# %%
tokenizer.train_from_iterator(batch_iterator(train_texts, batch_size=1000), trainer=trainer)
# %%
cls_token_id = tokenizer.token_to_id('[CLS]')
sep_token_id = tokenizer.token_to_id('[SEP]')
cls_token_id, sep_token_id
# %%
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)
# %%
encoding = tokenizer.encode("A single sequence")
encoding.tokens
# %%
encoding = tokenizer.encode("A sequence", "And its pair")
encoding.tokens
# %%
encoding.type_ids
# %%
tokenizer.decoder = decoders.WordPiece(prefix="##")
# %%
encoding = tokenizer.encode(train_texts[3])
# %%
train_texts[3]
# %%
encoding.tokens
# %%
tokenizer.encode('\t').tokens
# %%
tokenizer.enable_truncation(max_length=512)




# %% https://huggingface.co/blog/how-to-train

esperanto = load_dataset("oscar", "unshuffled_original_eo")




# %%  https://www.thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
from tokenizers import BertWordPieceTokenizer
# %%
tokenizer = BertWordPieceTokenizer()
# %%
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
# %%
vocab_size = 20000
max_length = 512
truncate_longer_samples = True
# %%
tokenizer.train_from_iterator(batch_iterator(train_texts, batch_size=1000), vocab_size=vocab_size,
    special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)
# %%
tokenizer.save_model(".data/wikitext2/")
# %%
with open(os.path.join(".data/wikitext2/", "config.json"), "w") as f:
  tokenizer_cfg = {
      "do_lower_case": True,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
  }
  json.dump(tokenizer_cfg, f)

# %%
BertTokenizerFast.from_pretrained(".data/wikitext2/")
# %%
