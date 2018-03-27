
default_config = {
        'dictionary_size': 20000,
        'random_seed': 0,
        'validation_size': 100,
        'max_length': 30,
}


def get_dataset(train_path, test_path, **config):
    """Reads and parses the dataset.

    Arguments:
        train_path: A string, the path to the file containing the training set.
        test_path: A string, the path to the file containing the test set.
        config: A configuration dictionary, providing e.g. `"dictionary_size"` or
            `"random_seed"`, `"validation_size"`, `"max_length"`.

    Returns:
        A tuple containing:
            The word dictionary mapping words and tokens (strings) to unique indices.
            The training set as an array of shape `[N, M]` where N is the set size and
                M is the maximum sentence length.
            The validation set (identical specs).
            The test set (identical specs).
    """

    raise NotImplementedError
