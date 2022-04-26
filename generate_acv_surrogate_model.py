import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, classification_report, roc_auc_score, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import sklearn

import sys
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

from acv_explainers import ACXplainer

import random

from tqdm import tqdm_notebook

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
cls_method = sys.argv[2]

classification = False if sys.argv[3]=="0" else True

random_state = 22
exp_iter = 10

save_to = "%s/%s/" % (PATH, dataset)
dataset_folder = "%s/datasets/" % (save_to)
final_folder = "%s/%s/" % (save_to, cls_method)

#Get datasets
X_train = pd.read_csv(dataset_folder+dataset+"_Xtrain.csv", index_col=False, sep = ";")
y_train = pd.read_csv(dataset_folder+dataset+"_Ytrain.csv", index_col=False, sep = ";")
test_x = pd.read_csv(final_folder+"test_sample.csv", index_col=False, sep = ";").values
results = pd.read_csv(os.path.join(final_folder,"results.csv"), index_col=False, sep = ";")
actual = results["Actual"].values

feat_list = [each.replace(' ','_') for each in X_train.columns]

cls = joblib.load(save_to+cls_method+"/cls.joblib")
scaler = joblib.load(save_to+"/scaler.joblib")

Y_pred = cls.predict(X_train)
test_pred = cls.predict(test_x)
kf = KFold(n_splits=5, shuffle = True, random_state=random_state)

space = {"n_estimators": scope.int(hp.quniform('n_estimators', 1, 20, q=1)),
        "max_depth": scope.int(hp.quniform('max_depth', 1, 20, q=1))}

trials = Trials()

if classification:
    def acv_classifier_optimisation(args, random_state = random_state, cv = kf, X = X_train.values, y = Y_pred):
        score = []

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator = ACXplainer(classifier = True, n_estimators = args["n_estimators"], max_depth = args['max_depth'])
            estimator.fit(X_train, y_train)

            score.append(roc_auc_score(y_test, estimator.predict(X_test)))
        
        score = np.mean(score)

        return -score

    best = fmin(acv_classifier_optimisation, space = space, algo=tpe.suggest, max_evals = 25, trials=trials, rstate = np.random.RandomState(random_state))
    explainer = ACXplainer(classifier = True, n_estimators = int(best['n_estimators']), max_depth = int(best['max_depth']))
    explainer.fit(X_train, Y_pred)

    print(classification_report(cls.predict(test_x), explainer.predict(test_x)))

else:

    def acv_regression_optimisation(args, random_state = random_state, cv = kf, X = X_train.values, y = Y_pred):
        score = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator = ACXplainer(classifier = False, n_estimators = args["n_estimators"], max_depth = args['max_depth'])
            estimator.fit(X_train, y_train)

            score.append(mean_absolute_percentage_error(y_test, estimator.predict(X_test)))
        
        score = np.mean(score)

        return score

    best = fmin(acv_regression_optimisation, space = space, algo=tpe.suggest, max_evals = 25, trials=trials, rstate = np.random.RandomState(random_state))
    explainer = ACXplainer(classifier = False, n_estimators = int(best['n_estimators']), max_depth = int(best['max_depth']))
    explainer.fit(X_train, Y_pred)

    print("MAPE:", mean_absolute_percentage_error(cls.predict(test_x), explainer.predict(test_x)))
    print("R-Squared:", r2_score(cls.predict(test_x), explainer.predict(test_x)))

joblib.dump(explainer, save_to+cls_method+"/acv_explainer_test.joblib")
