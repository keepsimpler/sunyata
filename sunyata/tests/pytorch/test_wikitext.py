from sunyata.pytorch.wikitext103 import WikiText103DataModule


def test_wikitext103():
    data_dir = "resources/wikitext-103-v1/"
    batch_size = 16
    vocab_size = 20000
    seq_len = 128

    wikitext103_datamodule = WikiText103DataModule(data_dir, batch_size, vocab_size, seq_len)