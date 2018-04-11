import collections
import itertools
import random

_default_config = {
        'dictionary_size': 20000,
        'validation_size': 100,
        'sentence_length': 30,
        'random_seed': 0,
}

_tokens = {
        'start': '<bos>',
        'end': '<eos>',
        'pad': '<pad>',
        'unknown': '<unk>',
}


def _parse_file(filename, config):
    with open(filename, 'r') as f:
        sentences = f.read().split('\n')
    sentences = [s.split() for s in sentences]  # split into words
    sentences = [[_tokens['start']] + s + [_tokens['end']] for s in sentences]
    sentences = [s for s in sentences if len(s) <= config['sentence_length']]  # filter
    sentences = [s + [_tokens['pad']]*(config['sentence_length']-len(s))
                 for s in sentences]  # pad
    return sentences


def _build_vocabulary(sentences, config):
    words = list(itertools.chain.from_iterable(sentences))
    counts = collections.Counter(words)
    words = sorted(counts.keys(), key=lambda w: (-counts[w], w))
    vocabulary = [_tokens['unknown']] + words[:config['dictionary_size']-1]
    return vocabulary


def _convert_words_to_indices(sentences, word2idx):
    return [[word2idx[w] if w in word2idx else word2idx[_tokens['unknown']] for w in s]
            for s in sentences]


def get_dataset(train_path, test_path, **user_config):
    """Reads and parses the dataset.

    Arguments:
        train_path: A string, the path to the file containing the training set.
        test_path: A string, the path to the file containing the test set.
        config: A configuration dictionary, providing e.g. `"dictionary_size"` or
            `"random_seed"`, `"validation_size"`, `"max_length"`.

    Returns:
        A tuple containing:
            The list of words (strings), where the index of each word corresponds to
                its encoding in the data.
            The training set as an array of shape `[N, M]` where N is the set size and
                M is the maximum sentence length.
            The validation set (identical specs).
            The test set (identical specs).
    """

    config = _default_config.copy()
    config.update(user_config)

    # Build vocabulary
    train_data = _parse_file(train_path, config)
    vocabulary = _build_vocabulary(train_data, config)
    word2idx = dict(zip(vocabulary, range(len(vocabulary))))

    # Convert to indices
    train_data = _convert_words_to_indices(train_data, word2idx)
    test_data = _convert_words_to_indices(_parse_file(test_path, config), word2idx)

    # Shuffle
    random.Random(config['random_seed']).shuffle(train_data)
    random.Random(config['random_seed']).shuffle(test_data)

    # Create validation split
    val_data = train_data[:config['validation_size']]
    train_data = train_data[config['validation_size']:]

    return vocabulary, train_data, val_data, test_data
