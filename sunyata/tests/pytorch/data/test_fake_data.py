from sunyata.pytorch.data.fake_data import yield_fake_data


def test_yield_fake_data():
    batch_size, seq_len, vocab_size, batchs_num = 2, 8, 16, 8
    fake_data = yield_fake_data(batch_size, seq_len, vocab_size, batchs_num)
    for i, (input, target) in enumerate(fake_data):
        pass
    
    assert i == batchs_num - 1
    assert input.shape == (batch_size, seq_len)
    assert target.shape == (batch_size, seq_len)

