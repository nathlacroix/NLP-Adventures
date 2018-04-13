from pathlib import Path
from dataset import get_dataset


if __name__ == '__main__':
    data_path = Path('data/')
    config = {
            'vocab_size': 20000,
            'validation_size': 100,
            'sentence_size': 30
    }

    vocab, train_data, val_data, test_data, pred_data = get_dataset(data_path, **config)

    assert len(vocab) == config['vocab_size']
    assert len(val_data) == config['validation_size']
    for data in [train_data, val_data, test_data]:
        for sentence in data:
            assert len(sentence) == config['sentence_size']
            assert vocab[sentence[0]] == '<bos>'
            assert vocab[sentence[-1]] in ['<eos>', '<pad>']

    for sentence in pred_data:
        assert len(sentence) < config['sentence_size']
        assert vocab[sentence[0]] == '<bos>'

    print('Dataset test passed.')
