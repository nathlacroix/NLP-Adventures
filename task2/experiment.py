import argparse
import os
import numpy as np
import csv
import yaml
import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from models.simple_logistic_regression import SimpleLogisticRegression
from models.svm import SimpleSVM
from models.xgboost import Xgboost
from models.mlp import SimpleMLP
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
np.warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.nan)



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
    """ Load the data and split in training and validation set. """
    features_path = Path(base_path, 'features/train')
    X = load_features(features_path, config['features'])
    X = X[0:1871]
    data_path = Path(base_path, 'data/val_only.csv')
    y = get_labels(data_path)
    return train_test_split(X, y,
                         test_size=config['validation_ratio'],
                          # test_size=.5, shuffle=False,
                         random_state=config['random_seed'])

def mask_stories_with_cond(X, y, cond=None):
    '''
    Takes X = features array, y = groundtruth, cond = condition array.
    :returns X_masked, y_masked, ind_masked'''
    if cond is None:
        cond = np.logical_and(np.logical_and(np.any(X[:, 0:4] != np.zeros((X.shape[0], 4)), axis=1),
                                         np.any(X[:, 4:8] != np.zeros((X.shape[0], 4)), axis=1)),
                          np.any(X[:, 0:4] != X[:, 4:8], axis=1))
    indices = np.arange(X.shape[0])
    X = X[cond]
    y = y[cond]
    indices = indices[cond]

    return X, y, indices


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

    models = {
        'lr': SimpleLogisticRegression(),
        'xgb': Xgboost(**config['model']['params']),
        'svm': SimpleSVM(),
        'mlp': SimpleMLP()
    }

    if config.get('eval_several_times', False):
        random_seeds = np.arange(25)

    else:
        random_seeds = [config['random_seed']]

    accuracies = []
    acc1 = []
    acc2 = []
    end1_is_0 = []
    end2_is_0 = []

    for rs in random_seeds:
        config['random_seed'] = rs
        # Load training data
        X_train, X_val, y_train, y_val = get_train_val_data(args.base_path, **config)
        #X_t_masked, y_t_masked, ind_train = mask_stories_with_cond(X_train, y_train, )
        #X_v_masked, y_v_masked, ind_val = mask_stories_with_cond(X_val, y_val)
        print(X_train.shape)
        #TOTAL
        # Train model

        model = models[config.get('model', {}).get('name', 'lr')]
        start = datetime.datetime.now()
        model.train(X_train[:,:36], y_train)
        print('Training time: {}' .format(datetime.datetime.now() - start))
        pred1 = model.predict(X_val[:,:36])
        proba_1 = model.predict_proba(X_val[:,:36])

        sent_model = models['lr']
        sent_model.train(X_train[:,36:], y_train)
        pred_sent = sent_model.predict(X_val[:,36:])

        offset = 0.10
        proba_comb = np.minimum(1, (offset + np.abs(proba_1[:, 0] - 0.5))*2)*pred1 + \
                     np.maximum(0, (1 - (offset + np.abs(proba_1[:, 0] - 0.5))*2))*pred_sent
        proba_mask = proba_comb > 1.5
        pred_comb = np.ones(X_val.shape[0], dtype=np.int)
        pred_comb[proba_mask] = 2

        pred_comb_check= np.ones(X_val.shape[0], dtype=np.int)
        pred_comb_check[proba_1[:,0] < .55] = pred_sent[proba_1[:,0] < .55]
        # Evaluate it
        #accuracy = model.score(X_val, y_val)
        accuracy = accuracy_score(y_val, pred_comb)
        print("Acc= " + str(accuracy))
        accuracies.append(accuracy)

        '''
        # Train model
        model = SimpleLogisticRegression()
        model.train(X_t_masked[:,16:], y_t_masked)
        pred1 = model.predict(X_v_masked[:,16:])

        # Evaluate it
        accuracy = model.score(X_v_masked[:,16:], y_v_masked)
        print("Acc1= " + str(accuracy))
        acc1.append(accuracy)
        
        mymodel = SimpleLogisticRegression()
        mymodel.train(X_t_masked[:,0:12],y_t_masked)
        pred2 = mymodel.predict(X_v_masked[:,0:12])

        # Evaluate it
        accuracy = mymodel.score(X_v_masked[:,0:12], y_v_masked)
        print("Acc2= " + str(accuracy))
        acc2.append(accuracy)

        i = 0
        pred_together = pred1
        for ind in ind_val:
            pred_together[ind] = pred2[i]
            i += 1
        
        acc = accuracy_score(y_val, pred_together)
        print('together: {}'.format(acc))
        
        if args.eval_several_times:
            accuracies.append(acc)
    '''
    # stats if several evals
    if config.get('eval_several_times', False):
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print("mean accuracy: {} +- {} ".format(mean, std))
        #print(model.get_weights())
        plt.boxplot(accuracies)
        title = 'features: {0}, mean accuracy: {1:.4f} +- {2:.4f}' \
            .format(config['features'], mean, std)
        plt.title(title)
        plt.show(block=False)
        plt.savefig('stats/' + str(config['features']) + str(config['model']['name']) +
                    datetime.datetime.now().strftime('%d_%m_%y_%H%M%S_') + '.png',
                    bbox_inches="tight")
    print(accuracies)
    print(ttest_rel(acc2, acc1))
    # Make a prediction if test is True
    if args.test:
        features_path = Path(args.base_path, 'features/test')
        X_test = load_features(features_path, config['features'])
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        # Write the prediction in a file
        output_path = Path(args.base_path, 'outputs')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filename = Path(output_path, args.exp_name + '.csv')
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for pred,prob in zip(y_pred, y_proba) :
                writer.writerow([str(pred)])
                #writer.writerow([str(prob)])
        print("Prediction stored in " + str(filename))
