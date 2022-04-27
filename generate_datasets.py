import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os
import sys
import joblib

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
balanced = sys.argv[2] #Is the dataset balanced?

random_state = 39

dataset_folder = "%s/%s/" % (PATH, dataset)

#Load and process data
dataset_name = "%s.csv" % (dataset)
dataset_path = dataset_folder + "/datasets/" + dataset_name
data = pd.read_csv( dataset_path )
#data = data.dropna()

# define target col and feature cols
class_cols = {"bike_sharing": "total_rental", "breast_cancer": "diagnosis", "compas": "high_risk",
              "diabetes": "Outcome", "facebook": "post_consumers", "housing": "MEDV", "income": "income", "iris": "class", 
              "mushroom": "target", "nursery": "class", "real_estate": "house_price", "solar_flare": "num_flares", 
              "student_scores": "G3", "wine_quality": "quality"}

class_var = class_cols[dataset]
feature_names = data.drop(class_var, axis=1).columns.to_list()

# balance dataset
if balanced == False:
    classes = data[class_var]
    neg_cases = data[data[class_var] == 0]
    pos_cases = data[data[class_var] == 1]

    if len(neg_cases) > len(pos_cases):
        neg_cases = neg_cases.sample(n=len(pos_cases), random_state = random_state)
    elif len(pos_cases) > len(neg_cases):
        pos_cases = pos_cases.sample(n=len(neg_cases), random_state = random_state)

    balanced_data = [neg_cases, pos_cases]
    balanced_data = pd.concat(balanced_data)

    # check how balanced the classes are
    print("Class balance:", balanced_data.groupby(class_var).count())
    
else:
    balanced_data = data

#Scale data and save scaler
X = balanced_data[ feature_names ]#.values
Y = balanced_data[class_var]#.values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns = X.columns)
joblib.dump(scaler, dataset_folder+"scaler.joblib")

#generate training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=515)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=515)

#Save datasets
X_train.to_csv(dataset_path.replace(".csv", "") + "_Xtrain.csv", sep=";", index = False)
X_test.to_csv(dataset_path.replace(".csv", "") + "_Xtest.csv", sep=";", index = False)
X_validation.to_csv(dataset_path.replace(".csv", "") + "_Xvalidation.csv", sep=";", index = False)

y_train.to_csv(dataset_path.replace(".csv", "") + "_Ytrain.csv", sep=";", index = False)
y_test.to_csv(dataset_path.replace(".csv", "") + "_Ytest.csv", sep=";", index = False)
y_validation.to_csv(dataset_path.replace(".csv", "") + "_Yvalidation.csv", sep=";", index = False)

#Create template to save results
results_template_test = pd.DataFrame(y_test)
results_template_validation = pd.DataFrame(y_validation)
results_template = pd.concat([results_template_test, results_template_validation])
results_template = results_template.rename(columns = {class_var: "Actual"})

results_template.to_csv(dataset_path.replace(".csv", "") + "_results_template.csv", sep=";", index = False)