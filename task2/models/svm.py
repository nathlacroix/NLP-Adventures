from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class SimpleSVM():
    def __init__(self):
        self.parameters_to_tune = {'C': [0.01, 0.1, 1., 10, 100]}
        self.nb_folds = 5
        self.clf = GridSearchCV(SVC(kernel='linear', probability=True), self.parameters_to_tune,
                                cv=self.nb_folds, scoring='accuracy')

    def train(self, X_train, y_train):
        print("Start training...")
        self.clf.fit(X_train, y_train)
        print("Training is over.")

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def predict_proba(self, X_test):
        return self.clf.predict_proba(X_test)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    def get_weights(self):
        return self.clf.best_estimator_.coef_