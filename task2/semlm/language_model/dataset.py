import collections
import itertools
import random
import csv
from pathlib import Path

_default_config = {
        'vocab_size': 20000,
        'validation_size': 5000,
        'nb_frames': 0,
        'random_seed': 0,
        'eval_prefix': 'sct_val_framenet',
        'training_file': 'sct_train_framenet.txt',
}

tokens = {'pad': '<pad>', 'unknown': '<unk>'}


def _parse_file(filename, config, pad=True):
    sentences = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            # Split the line if needed
            if config['nb_frames']:
                frames = [row[i:i+config['nb_frames']]
                          for i in range(0, len(row), config['nb_frames'])]
            else:
                frames = [row]
                sentences += frames
    if pad:
        sentence_size = max([len(s) for s in sentences])
        sentences = [s + [tokens['pad']] * (sentence_size - len(s))
                     for s in sentences]  # pad
    return sentences


def _parse_test_files(filenames, config):
    context = _parse_file(filenames[0], config, pad=False)
    ending1 = _parse_file(filenames[1], config, pad=False)
    ending2 = _parse_file(filenames[2], config, pad=False)
    return [context, ending1, ending2]


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
            The list of componants (i.e. a list of strings), where the
                index of each component corresponds to its encoding in the data.
            The training set as a list of list of components (e.g. a list of stories)
            The validation set (identical specs).
            The test data: a list of the form [context, ending1, ending2],
                where each element is a list of list of components
    """

    config = _default_config.copy()
    config.update(user_config)

    train_path = Path(data_path, config['training_file'])

    # Build vocabulary
    train_data = _parse_file(train_path, config)
    vocabulary = _build_vocabulary(train_data, config)
    word2idx = dict(zip(vocabulary, range(len(vocabulary))))

    # Split the training set to get a validation set and prepare the test set
    train_data = _convert_words_to_indices(train_data, word2idx)
    random.Random(config['random_seed']).shuffle(train_data)
    val_data = train_data[:config['validation_size']]
    train_data = train_data[config['validation_size']:]
    test_context_path = Path(data_path, config['eval_prefix'] + '.txt')
    test_ending1_path = Path(data_path, config['eval_prefix'] + '_ending0.txt')
    test_ending2_path = Path(data_path, config['eval_prefix'] + '_ending1.txt')
    test_data = _parse_test_files([test_context_path,
                                   test_ending1_path,
                                   test_ending2_path], config)
    test_data = [_convert_words_to_indices(d, word2idx) for d in test_data]

    return vocabulary, train_data, val_data, test_data
