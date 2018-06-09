import argparse
import os
import numpy as np
import csv
import yaml
from pathlib import Path
from models.simple_logistic_regression import SimpleLogisticRegression


def get_labels(file_name):
    """ Get the correct labels from the file 'file_name'. """
    if(not file_name.exists()):
        raise Exception(str(file_name) + " cannot be found. Aborting.")
    labels = []
    with open(str(file_name), 'r') as csvfile:
        csvfile.readline()  # get rid of the header
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(int(row[7]))
    return np.array(labels)


def load_features(path, features):
    """ Load previously computed features.

    Arguments:
        path: path of the directory containing the features stored as .npz
        features: list of features to use

    Output: a np array concatenating all the features.
    """
    arrays = []
    for feat in features:
        data_path = Path(path, feat + '.npz')
        if not data_path.exists():
            print(str(data_path) + " cannot be found, skipping it.")
            continue
        with np.load(data_path) as data:
            feat_array = [data[key] for key in data.files]

            # check if arrays are one dimensional (=1 array = 1 feature)
            #  or if arrays contain several features and stack accordingly
            if(len(feat_array[0].shape)) == 1:
                feat_array = np.stack(feat_array, axis=1)
                arrays.append(feat_array)
            else:
                arrays.append(np.hstack(feat_array))
    if len(arrays) == 0:
        raise Exception("No features found, aborting.")
    return np.concatenate(arrays, axis=1)


def get_train_val_data(base_path, **config):
    """ Load the training and validation data. """
    # Load training data
    train_features_path = Path(base_path, 'features/val')
    X_train = load_features(train_features_path, config['features'])
    train_data_file = Path(base_path, 'data/' + config['val_labels_file'])
    y_train = get_labels(train_data_file)

    # Load validation data
    val_features_path = Path(base_path, 'features/test')
    X_val = load_features(val_features_path, config['features'])
    val_data_file = Path(base_path, 'data/' + config['test_labels_file'])
    y_val = get_labels(val_data_file)
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str,
                        help="path to folder containing the data, features and outputs")
    parser.add_argument('config', type=str, help="path to config file")
    parser.add_argument('--exp_name', type=str, default='output', help="experiment name")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # Load training data
    X_train, X_val, y_train, y_val = get_train_val_data(args.base_path, **config)

    # Train model
    model = SimpleLogisticRegression()
    model.train(X_train, y_train)

    # Evaluate it
    accuracy = model.score(X_val, y_val)
    print("Validation accuracy = " + str(accuracy))

    # Make a prediction if test is True
    if args.test:
        features_path = Path(args.base_path, 'features/eth_test')
        X_test = load_features(features_path, config['features'])
        y_pred = model.predict(X_test)

        # Write the prediction in a file
        output_path = Path(args.base_path, 'outputs')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = Path(output_path, args.exp_name + '.csv')
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for pred in y_pred:
                writer.writerow([str(pred)])
        print("Prediction stored in " + str(filename))
