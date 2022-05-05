from lib2to3.pgen2 import token
import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

import pytorch_lightning as pl


class WikiText103DataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str, batch_size: int, vocab_size: int, seq_len:int):
        super().__init__()
        self.data_dir, self.batch_size, self.vocab_size, self.seq_len = data_dir, batch_size, vocab_size, seq_len
        self.setup()
    
    def setup(self, stage=None):
        train_tensor_file = f"{self.data_dir}wikitext-103-vocab{self.vocab_size}-seq{self.seq_len}.train.tensor"
        validation_tensor_file = f"{self.data_dir}wikitext-103-vocab{self.vocab_size}-seq{self.seq_len}.validation.tensor"
        tokenizer_path = f"{self.data_dir}wikitext-103-vocab{self.vocab_size}/"

        if os.path.exists(train_tensor_file) and os.path.exists(validation_tensor_file) and os.path.exists(tokenizer_path):
            self.train_data = torch.load(train_tensor_file)
            self.validation_data = torch.load(validation_tensor_file)
            self.tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_path + "vocab.json", tokenizer_path + "merges.txt")
        else:
            self.train_data, self.validation_data, self.tokenizer = setup_wikitext103(self.data_dir, self.vocab_size, self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=shift_one_token
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=shift_one_token
        )


def shift_one_token(batch):
    batch = torch.stack(batch)
    input = batch[:, :-1]
    target = batch[:, 1:]
    return input, target


def setup_wikitext103(data_dir: str, vocab_size: int, seq_len: int):
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(batch_iterator(dataset, "train"), vocab_size=vocab_size,
        min_frequency=2, special_tokens=["<unk>","\n"])
    tokenizer_path = f"{data_dir}wikitext-103-vocab{vocab_size}/"
    if not os.path.exists(tokenizer_path):
        os.mkdir(tokenizer_path)
    tokenizer.save_model(tokenizer_path)

    train_data = tokenizer.encode_batch([line for line in dataset['train']['text'] if len(line) > 0]) 
    train_ids = to_tensor(train_data, seq_len)
    torch.save(train_ids, f"{data_dir}wikitext-103-vocab{vocab_size}-seq{seq_len}.train.tensor")

    validation_data = tokenizer.encode_batch([line for line in dataset['validation']['text'] if len(line) > 0]) 
    validation_ids = to_tensor(validation_data, seq_len)
    torch.save(validation_ids, f"{data_dir}wikitext-103-vocab{vocab_size}-seq{seq_len}.validation.tensor")
    return train_ids, validation_ids, tokenizer


def batch_iterator(dataset, split: str, batch_size=1000):
    for i in range(0, len(dataset[split]), batch_size):
        yield dataset[split][i: i + batch_size]["text"]


def to_tensor(data: list, seq_len: int) -> torch.tensor:
    ids = []
    for line in data:
        ids.extend(line.ids)
    num_ids = len(ids) // seq_len * seq_len
    ids = ids[:num_ids]
    ids = torch.tensor(ids, dtype=torch.long).view(-1, seq_len)
    return ids
