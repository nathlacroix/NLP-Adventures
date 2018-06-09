from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np

class SimpleMLP():
    def __init__(self):
        self.parameters_to_tune = {'alpha': [5, 15, 20, 30, 50,100],
                                   'hidden_layer_sizes':  [(100,)],
                                   'solver': ['lbfgs']}
        self.nb_folds = 5
        self.clf = GridSearchCV(MLPClassifier(), self.parameters_to_tune,
                                cv=self.nb_folds, scoring='accuracy')

    def train(self, X_train, y_train):
        print("Start training...")
        self.clf.fit(X_train, y_train)
        print("Training is over. Best params: {}" .format(self.clf.best_params_))
        print("CV results for params: {} : \n {} \n {} " .format(self.clf.cv_results_['params'],
                                                                 self.clf.cv_results_['mean_train_score'],
                                                                 self.clf.cv_results_['mean_test_score']))

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    def get_weights(self):
        return self.clf.best_estimator_.coefs_