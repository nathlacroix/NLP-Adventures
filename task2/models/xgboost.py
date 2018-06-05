from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from scipy import stats as st

one_to_left = st.beta(5, 1)
from_zero_positive = st.expon(0, 50)
params = {
    "n_estimators": st.randint(50, 200),
    "max_depth": st.randint(3, 10),
    "learning_rate": st.uniform(0.1, 0.4),
    "colsample_bytree": one_to_left,
    "gamma": st.uniform(5, 10),
    'reg_lambda': st.uniform(0,15),
}

class Xgboost():
    def __init__(self, **extra_params):
        self.parameters_to_tune = {'max_depth': [ 5],
                                   'learning_rate': [ .1, .3],
                                   'n_estimators': [100]}
        self.nb_folds = 5
        #self.clf = GridSearchCV(XGBClassifier(n_jobs=-1 ), self.parameters_to_tune,
                               # cv=self.nb_folds, scoring='accuracy')
        self.clf = RandomizedSearchCV(XGBClassifier(n_jobs=-1 ), params, n_iter=extra_params.get('n_iter', 1),
                                cv=self.nb_folds, scoring='accuracy')


    def train(self, X_train, y_train):
        print("Start training...")
        self.clf.fit(X_train, y_train)
        print("Training is over.")
        print("Best params: {}" .format(self.clf.best_params_))
        print(self.clf.cv_results_)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def score(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    def get_weights(self):
        return self.clf.best_estimator_.coef_