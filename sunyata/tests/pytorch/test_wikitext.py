from sunyata.pytorch.wikitext import WikiTextDataModule


def test_wikitext103():
    data_dir = ".data/wikitext/"
    batch_size = 16
    vocab_size = 20000
    seq_len = 128

    wikitext103_datamodule = WikiTextDataModule("103", data_dir, batch_size, vocab_size, seq_len)