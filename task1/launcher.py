import yaml
import os
import argparse
import numpy as np
import tensorflow as tf

from dataset import get_dataset
from experiment import train, eval, pred


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


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


def _train(data_path, exp_name, config):
    exp_dir = _initialize_structure(data_path, exp_name, config)
    set_seed(config['random_seed'])
    datasets = get_dataset(data_path, **config)
    train_data = np.array(datasets[1])
    val_data = np.array(datasets[2])
    train(train_data, val_data, exp_dir, **config)


def _eval(data_path, exp_name, config):
    exp_dir = _initialize_structure(data_path, exp_name, config)
    if not os.path.exists(os.path.join(exp_dir,
                                       "models/model.ckpt.index")):
        raise Exception("The model for experiment "
                        + exp_name +
                        " has to be trained first!" +
                        " Please train the model before evaluating it.")

    set_seed(config['random_seed'])
    datasets = get_dataset(data_path, **config)
    test_data = np.array(datasets[3])
    eval(test_data, exp_dir, **config)


def _pred(data_path, exp_name, config):
    exp_dir = _initialize_structure(data_path, exp_name, config)
    if not os.path.exists(os.path.join(exp_dir,
                                       "models/model.ckpt.index")):
        raise Exception("The model for experiment "
                        + exp_name +
                        " has to be trained first!" +
                        " Please train the model before doing a prediction.")

    set_seed(config['random_seed'])
    datasets = get_dataset(data_path, **config)
    dictionary = datasets[0]
    pred_data = np.array(datasets[4])
    pred(pred_data, dictionary, exp_dir, **config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--config', type=str, help="path to config file")
    p_train.add_argument('--data', type=str, help="path to data folder")
    p_train.add_argument('--exp', type=str, help="name of the experiment")
    p_train.set_defaults(func=_train)

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('--config', type=str, help="path to config file")
    p_train.add_argument('--data', type=str, help="path to data folder")
    p_train.add_argument('--exp', type=str, help="name of the experiment")
    p_train.set_defaults(func=_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('--config', type=str, help="path to config file")
    p_train.add_argument('--data', type=str, help="path to data folder")
    p_train.add_argument('--exp', type=str, help="name of the experiment")
    p_train.set_defaults(func=_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    args.func(args.data, args.exp, config)
