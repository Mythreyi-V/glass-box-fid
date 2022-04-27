import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB

import os
import sys
import joblib

import warnings
warnings.filterwarnings('ignore')

import random

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
cls_method = sys.argv[2]

classification = False if sys.argv[3]=="0" else True

random_state = 39
num_eval = 500
n_splits = 3
random.seed(random_state)

save_to = "%s/%s/" % (PATH, dataset)
dataset_folder = "%s/datasets/" % (save_to)

#Get datasets
X_train = pd.read_csv(dataset_folder+dataset+"_Xtrain.csv", index_col=False, sep = ";")#.values
X_test = pd.read_csv(dataset_folder+dataset+"_Xtest.csv", index_col=False, sep = ";")#.values
X_validation = pd.read_csv(dataset_folder+dataset+"_Xvalidation.csv", index_col=False, sep = ";")#.values

y_train = pd.read_csv(dataset_folder+dataset+"_Ytrain.csv", index_col=False, sep = ";").values.reshape(-1)
y_test = pd.read_csv(dataset_folder+dataset+"_Ytest.csv", index_col=False, sep = ";").values.reshape(-1)
y_validation = pd.read_csv(dataset_folder+dataset+"_Yvalidation.csv", index_col=False, sep = ";").values.reshape(-1)

feat_list = X_train.columns
results_template = pd.read_csv(os.path.join(dataset_folder, dataset+"_results_template.csv"), index_col=False)

#Set hyperparameter grid
if cls_method == "decision_tree":
    space = {"splitter": ["best", "random"],
            "min_samples_split": [random.uniform(0, 1) for i in range (50)],
            "max_features": [random.uniform(0,1) for i in range (50)]}
    fit_params = {"sample_weight": None}

elif cls_method == "logit":
    space = {"fit_intercept": [True, False],
             "penalty": ['l1', 'l2', 'elasticnet', 'none'],
             "max_iter": [random.uniform(5,200) for i in range (50)],
             "tol": np.logspace(-4, 4, 50)}
    fit_params = {"sample_weight": None}

elif cls_method == "lin_reg":
    space = {"fit_intercept": [True, False]}
    fit_params = {"sample_weight": None}    

elif cls_method == "nb":
    space = {'var_smoothing': np.logspace(0, -9, 100)}
    fit_params = {}

    
#Create and train model
if classification == True:
    if cls_method == "decision_tree":
        space["criterion"] = ["gini", "entropy"]
        estimator = DecisionTreeClassifier(random_state = random_state)
    elif cls_method == "logit":
        estimator = LogisticRegression(random_state = random_state)
    elif cls_method == "nb":
        estimator = GaussianNB()
else:
    if cls_method == "decision_tree":
        space["criterion"] = ["mse", "friedman_mse", "mae", "poisson"]
        estimator = DecisionTreeRegressor(random_state = random_state)
    elif cls_method == "lin_reg":
        estimator = LinearRegression()
        
cls = GridSearchCV(estimator, space, verbose = 1)
cls.fit(X_train.values, y_train, **fit_params)

cls = cls.best_estimator_
joblib.dump(cls, save_to+cls_method+"/cls.joblib")

#Test model accuracy
test_x = pd.concat([X_test, X_validation])
test_y = np.hstack([y_test, y_validation])
y_pred = cls.predict(test_x.values)

if classification == True:
    print(classification_report(test_y, y_pred))
else:
    print("RMSE:", mean_squared_error(test_y, y_pred, squared = False))
    print("MAE:", mean_absolute_error(test_y, y_pred))
    print("MAPE:", mean_absolute_percentage_error(test_y, y_pred))
    
if classification:
    full_test = pd.concat([test_x.reset_index(), results_template], axis = 1, join = 'inner').drop(['index'], axis = 1)
    full_test["predicted"] = y_pred
    
    grouped = full_test.groupby('predicted')
    if grouped.size().min() <= 50:
      balanced = grouped.apply(lambda x: x.sample(grouped.size().min()).reset_index(drop=True))
    else:
      balanced = grouped.apply(lambda x: x.sample(250).reset_index(drop=True))
    
    test_sample = balanced[X_test.columns]
    test_sample.reset_index(drop = True, inplace = True)
    
    results_template = balanced[results_template.columns]
    results_template.reset_index(drop = True, inplace = True)
    
    if cls_method == "brl":
        preds = cls.predict(test_sample.values, threshold = 0.5)
    else:
        preds = cls.predict(test_sample.values)
    probas = [cls.predict_proba(test_sample.values)[i][preds[i]] for i in range(len(preds))]

    results_template["Prediction"] = preds
    results_template["Prediction Probability"] = probas
    
else:
    full_test = pd.concat([test_x.reset_index(), results_template], axis = 1, join = 'inner').drop(['index'], axis = 1)
    if len(full_test) <= 100:
      sample = full_test
    else:
      sample = full_test.sample(500).reset_index(drop=True)

    test_sample = sample[X_test.columns]
    test_sample.reset_index(drop = True, inplace = True)

    results_template = sample[results_template.columns]
    results_template.reset_index(drop = True, inplace = True)
    
    preds = cls.predict(test_sample.values)
    results_template["Prediction"] = preds
    
results_template.to_csv(os.path.join(save_to, cls_method, "results.csv"), sep = ";", index = False)
test_sample.to_csv(os.path.join(save_to, cls_method, "test_sample.csv"), sep = ";", index = False)
