import yaml
import os
import argparse
from gensim import models
import numpy as np

from dataset import get_dataset
from experiment import train, eval, pred


def _initialize_structure(data_path, exp_name, config):
    if not os.path.exists(data_path):
        raise Exception("Wrong data path: no such file or directory.")
    exp_dir = os.path.join(config['log_dir'], "exp_" + exp_name)
    models_dir = os.path.join(exp_dir, "models")
    output_dir = os.path.join(exp_dir, "outputs")
    summaries_dir = os.path.join(exp_dir, "summaries")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)
    return exp_dir


def load_embedding(data_path, dictionary, config):
    """ Load an external word embedding stored in data_path.

    Arguments:
        data_path: path to the data where the word embedding has to be stored.
        dictionary: a list of all the words.

    Returns: a np array of size (embedding_size, vocab_size)
    """
    path = os.path.join(data_path, config['embedding_name'])
    if not os.path.exists(path):
        raise Exception("Pretrained word embedding not found." +
                        " Please add the file " + config['embedding_name'] +
                        " to your data path.")

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(config['vocab_size'], config['embedding_size']))

    for idx, tok in enumerate(dictionary):
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25,
                                                        high=0.25,
                                                        size=config['embedding_size'])
    return external_embedding


def _train(data_path, exp_name, config):
    exp_dir = _initialize_structure(data_path, exp_name, config)
    datasets = get_dataset(data_path, **config)
    dictionary = datasets[0]
    train_data = np.array(datasets[1])
    val_data = np.array(datasets[2])

    # Load an external embedding if necessary
    embedding = None
    if config['use_external_embedding']:
        embedding = load_embedding(data_path, dictionary, config)

    # Get the index corresponding to token <pad>
    pad_token = "<pad>"
    if pad_token in dictionary:
        pad_ind = dictionary.index(pad_token)
    else:
        raise Exception("Token <pad> should be present in dictionary.")

    train(train_data, val_data, embedding, pad_ind, exp_dir, **config)


def _eval(data_path, exp_name, config):
    exp_dir = _initialize_structure(data_path, exp_name, config)
    if not os.path.exists(os.path.join(exp_dir,
                                       "models/model.ckpt.index")):
        raise Exception("The model for experiment "
                        + exp_name +
                        " has to be trained first!" +
                        " Please train the model before evaluating it.")

    datasets = get_dataset(data_path, **config)
    dictionary = datasets[0]
    test_data = np.array(datasets[3])

    # Get the index corresponding to token <pad>
    pad_token = "<pad>"
    if pad_token in dictionary:
        pad_ind = dictionary.index(pad_token)
    else:
        raise Exception("Token <pad> should be present in dictionary.")

    eval(test_data, pad_ind, exp_dir, **config)


def _pred(data_path, exp_name, config):
    exp_dir = _initialize_structure(data_path, exp_name, config)
    if not os.path.exists(os.path.join(exp_dir,
                                       "models/model.ckpt.index")):
        raise Exception("The model for experiment "
                        + exp_name +
                        " has to be trained first!" +
                        " Please train the model before doing a prediction.")

    datasets = get_dataset(data_path, **config)
    dictionary = datasets[0]
    pred_data = datasets[4]

    # Get the index corresponding to token <pad>
    pad_token = "<pad>"
    if pad_token in dictionary:
        pad_ind = dictionary.index(pad_token)
    else:
        raise Exception("Token <pad> should be present in dictionary.")

    pred(pred_data, dictionary, pad_ind, exp_dir, **config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str, help="path to config file")
    p_train.add_argument('data', type=str, help="path to data folder")
    p_train.add_argument('exp', type=str, help="name of the experiment")
    p_train.set_defaults(func=_train)

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('config', type=str, help="path to config file")
    p_train.add_argument('data', type=str, help="path to data folder")
    p_train.add_argument('exp', type=str, help="name of the experiment")
    p_train.set_defaults(func=_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('config', type=str, help="path to config file")
    p_train.add_argument('data', type=str, help="path to data folder")
    p_train.add_argument('exp', type=str, help="name of the experiment")
    p_train.set_defaults(func=_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    args.func(args.data, args.exp, config)
