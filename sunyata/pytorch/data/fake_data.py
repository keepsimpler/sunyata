import torch
from torch.utils.data import Dataset
from typing import Tuple, Iterable

class FakeLMDataset(Dataset):
    def __init__(self, sequences_num: int, vocab_size: int, seq_len: int):
        torch.manual_seed(1)
        self.data = torch.randint(0, vocab_size-1, (sequences_num, seq_len+1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input = self.data[index, :-1]
        target = self.data[index, 1:]
        return input, target


def yield_fake_data(batch_size, seq_len, vocab_size, batchs_num) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(1)
    for _ in range(batchs_num):
        batch = torch.randint(0, vocab_size-1, (batch_size, seq_len+1))
        input = batch[:, :-1]
        target = batch[:, 1:]
        yield input, target

