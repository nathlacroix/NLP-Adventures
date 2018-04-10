import yaml
import os
import argparse
import numpy as np
import tensorflow as tf

from dataset import get_dataset
from model import train, eval, pred


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def _train(data_path, experiment, config):
    set_seed(config['random_seed'])
    datasets = get_dataset(os.path.join(data_path, 'sentences.train'),
                           os.path.join(data_path, 'sentences.eval'),
                           **config)
    data_train = datasets[1:3]
    train(data_train, experiment, **config)


def _eval(data_path, experiment, config):
    if not os.path.exists(os.path.join(config['saved_models_dir'],
                                       'experiment_' + experiment,
                                       "experiment_" + experiment + ".ckpt.index")):
        raise Exception("The model for experiment "
                        + experiment +
                        " has to be trained first!" +
                        " Please train the model before evaluating it.")

    set_seed(config['random_seed'])
    datasets = get_dataset(os.path.join(data_path, 'sentences.train'),
                           os.path.join(data_path, 'sentences.eval'),
                           **config)
    data_eval = datasets[3]
    eval(data_eval, experiment, **config)


def _pred(data_path, experiment, config):
    if not os.path.exists(os.path.join(config['saved_models_dir'],
                                       'experiment_' + experiment,
                                       "experiment_" + experiment + ".ckpt.index")):
        raise Exception("The model for experiment "
                        + experiment +
                        " has to be trained first!" +
                        " Please train the model before making a prediction.")

    set_seed(config['random_seed'])
    datasets = get_dataset(os.path.join(data_path, 'sentences.train'),
                           os.path.join(data_path, 'sentences.continuation'),
                           **config)
    dictionary = datasets[0]
    data_pred = datasets[3]
    pred(data_pred, dictionary, experiment, **config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--config', type=str, help="path to config file")
    p_train.add_argument('--data', type=str, help="path to data folder")
    p_train.add_argument('--exp', type=str, help="experiment ('A', 'B' or 'C')")
    p_train.set_defaults(func=_train)

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('--config', type=str, help="path to config file")
    p_train.add_argument('--data', type=str, help="path to data folder")
    p_train.add_argument('--exp', type=str, help="experiment ('A', 'B' or 'C')")
    p_train.set_defaults(func=_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('--config', type=str, help="path to config file")
    p_train.add_argument('--data', type=str, help="path to data folder")
    p_train.add_argument('--exp', type=str, default='C',
                         help="experiment ('A', 'B' or 'C')")
    p_train.set_defaults(func=_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    args.func(args.data, args.exp, config)
