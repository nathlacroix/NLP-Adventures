from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import numpy as np

class SimpleMLP():
    def __init__(self):
        self.parameters_to_tune = {'alpha': 10.0 ** - np.arange(-3, 3)}
        self.nb_folds = 5
        self.clf = GridSearchCV(MLPClassifier(hidden_layer_sizes=(40,20), solver='adam'), self.parameters_to_tune,
                                cv=self.nb_folds, scoring='accuracy')

    def train(self, X_train, y_train):
        print("Start training...")
        self.clf.fit(X_train, y_train)
        print("Training is over. Best params: {}" .format(self.clf.best_params_))
        print("CV results : {} " .format(self.clf.cv_results_))

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    def get_weights(self):
        return self.clf.best_estimator_.coefs_