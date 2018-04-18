import collections
import itertools
from pathlib import Path

_default_config = {
        'vocab_size': 20000,
        'validation_size': 100,
        'sentence_size': 30,
        'random_seed': 0,
        'eval_file': 'sentences.test',
}

tokens = {
        'start': '<bos>',
        'end': '<eos>',
        'pad': '<pad>',
        'unknown': '<unk>',
}


def _parse_file(filename, config, add_end_token=True, pad=True):
    with open(filename, 'r') as f:
        sentences = f.read().split('\n')
    sentences = [s.split() for s in sentences]  # split into words
    sentences = [[tokens['start']] + s for s in sentences]
    if add_end_token:
        sentences = [s + [tokens['end']] for s in sentences]
    sentences = [s for s in sentences if len(s) <= config['sentence_size']]  # filter
    if pad:
        sentences = [s + [tokens['pad']]*(config['sentence_size']-len(s))
                     for s in sentences]  # pad
    return sentences


def _build_vocabulary(sentences, config):
    words = list(itertools.chain.from_iterable(sentences))
    counts = collections.Counter(words)
    words = sorted(counts.keys(), key=lambda w: (-counts[w], w))
    vocabulary = [tokens['unknown']] + words[:config['vocab_size']-1]
    return vocabulary


def _convert_words_to_indices(sentences, word2idx):
    return [[word2idx[w] if w in word2idx else word2idx[tokens['unknown']] for w in s]
            for s in sentences]


def get_dataset(data_path, **user_config):
    """Reads and parses the dataset.

    Arguments:
        data_path: A string, the path of the root directory containing the dataset.
        config: A configuration dictionary, providing e.g. `"vocab_size"` or
            `"random_seed"`, `"validation_size"`, `"max_length"`.

    Returns:
        A tuple containing:
            The list of words (strings), where the index of each word corresponds to
                its encoding in the data.
            The training set as an array of shape `[N, M]` where N is the set size and
                M is the maximum sentence length.
            The validation set (identical specs).
            The test set (identical specs).
            The prediction set (identical specs) used for conditional generation.
    """

    config = _default_config.copy()
    config.update(user_config)

    train_path = Path(data_path, 'sentences.train')
    val_path = Path(data_path, 'sentences.eval')
    test_path = Path(data_path, config['eval_file'])
    pred_path = Path(data_path, 'sentences.continuation')

    # Build vocabulary
    train_data = _parse_file(train_path, config)
    vocabulary = _build_vocabulary(train_data, config)
    word2idx = dict(zip(vocabulary, range(len(vocabulary))))

    # Convert to indices
    train_data = _convert_words_to_indices(train_data, word2idx)
    val_data = _convert_words_to_indices(_parse_file(val_path, config), word2idx)
    test_data = _convert_words_to_indices(_parse_file(test_path, config), word2idx)
    pred_data = _convert_words_to_indices(
            _parse_file(pred_path, config, add_end_token=False, pad=False), word2idx)

    return vocabulary, train_data, val_data, test_data, pred_data
