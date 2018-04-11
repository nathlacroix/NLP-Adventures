from pathlib import Path
from dataset import get_dataset


if __name__ == '__main__':
    base_path = Path('data/')
    train_path = Path(base_path, 'sentences.train')
    test_path = Path(base_path, 'sentences.eval')

    config = {
            'dictionary_size': 20000,
            'validation_size': 100,
            'setence_length': 30
    }

    vocab, train_data, val_data, test_data = get_dataset(
            train_path, test_path, **config)

    assert len(vocab) == 20000
    assert len(val_data) == 100
    for data in [train_data, val_data, test_data]:
        for sentence in data:
            assert len(sentence) == 30
            assert vocab[sentence[0]] == '<bos>'
            assert vocab[sentence[-1]] in ['<eos>', '<pad>']

    print('Dataset test passed.')
