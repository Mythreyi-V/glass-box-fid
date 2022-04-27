import pandas as pd
import numpy as np

import sklearn

import os
import sys
import joblib

import warnings
warnings.filterwarnings('ignore')

import scipy

import lime
import lime.lime_tabular
import shap

from learning import *
import pyAgrum

from acv_explainers import ACXplainer


import seaborn as sns

import random

from tqdm import tqdm

import json

#Get true features for lin_reg and logit
def get_reg_features(cls):
    
    og_coef = cls.coef_
    if len(og_coef.shape) > 1:
        og_coef = og_coef[0]
    
    coef = [abs(val) for val in og_coef]
    
    bins = pd.cut(coef, 4, retbins = True, duplicates = "drop")
    q1_min = bins[1][-2]
    
    feat_pos = [i for i in range(len(coef)) if coef[i] > q1_min]
    
    return feat_pos

#Get true features for nb
def get_nb_features(cls, instance):
    pred = cls.predict(instance.reshape(1, -1))
    means = cls.theta_[pred][0]
    std = np.sqrt(cls.var_[pred])[0]

    likelihoods = []
    
    for i in range(len(means)):
        lkhood = scipy.stats.norm(means[i], std[i]).logpdf(instance[i])
        likelihoods.append(lkhood)
    
    bins = pd.cut(likelihoods, 4, retbins = True, duplicates = "drop")[1]
    lim = bins[-2]

    feat_pos = [i for i in range(len(likelihoods)) if likelihoods[i] >= lim]

    return feat_pos

#Get true features for dt
def get_tree_features(cls, instance):
    tree = cls.tree_
    lvl = 0
    left_child = tree.children_left[lvl]
    right_child = tree.children_right[lvl]

    feats = []
    
    while left_child != sklearn.tree._tree.TREE_LEAF and right_child != sklearn.tree._tree.TREE_LEAF:
        feature = tree.feature[lvl]
        feats.append(feature)
        
        if instance[feature] < tree.threshold[lvl]:
            lvl = left_child
        else:
            lvl = right_child
            
        left_child = tree.children_left[lvl]
        right_child = tree.children_right[lvl]
            
            
    feat_pos = set(feats)
    
    return feat_pos

#Feature extraction from SHAP
def get_shap_features(explainer, instance, cls, classification, exp_iter, feat_list):
    
    shap_exp = []
    pred = cls.predict(instance.reshape(1, -1))
    for i in range(exp_iter):
        if type(explainer) == shap.explainers._tree.Tree:
            exp = explainer(instance, check_additivity = False).values
        else:
            exp = explainer(instance.reshape(1, -1)).values
        
        if exp.shape == (1, len(feat_list), 2):
            exp = exp[0]
        
        if exp.shape == (len(feat_list), 2):
            exp = np.array([feat[pred] for feat in exp]).reshape(len(feat_list))
        elif exp.shape == (1, len(feat_list)) or exp.shape == (len(feat_list), 1):
            exp = exp.reshape(len(feat_list))
                        
        shap_exp.append(exp)
                
    if np.array(shap_exp).shape != (exp_iter, len(feat_list)):
        raise Exception("Explanation shape is not correct. It is", np.array(shap_exp).shape, "instead of the expected", (exp_iter, len(feat_list)))
            
    avg_val = np.average(shap_exp, axis = 0)
    abs_val = [abs(val) for val in avg_val]
    
    #Get recall and precision for the average of shap values
    bins = pd.cut(abs_val, 4, retbins = True, duplicates = "drop")
    q1_min = bins[1][-2]

    sorted_val = np.copy(abs_val)
    sorted_val.sort()
    
    shap_features = [i for i in range(len(feat_list)) if abs_val[i] > q1_min]
    
    return shap_features

#Feature extraction from LIME
def get_lime_features(explainer, instance, cls, classification, exp_iter, feat_list):
    lime_exp = []
    
    for i in range(exp_iter):
        if classification==True:
            lime_exp.extend(explainer.explain_instance(instance, cls.predict_proba, 
                                                num_features=len(feat_list), labels=[0,1]).as_list())
        else:
            lime_exp.extend(explainer.explain_instance(instance, cls.predict, 
                                                num_features=len(feat_list), labels=[0,1]).as_list())
            
    weights = [[] for each in feat_list]
    for exp in lime_exp:
        feat = exp[0]
        if '<' in feat:
            feat = exp[0].replace("= ",'')
            parts = feat.split('<')
        elif '>' in feat:
            feat = exp[0].replace("= ",'')
            parts = feat.split('>')
        else:
            parts = feat.split("=")
        
        for part in parts:
            if part.replace('.','').replace(' ','').isdigit()==False:
                feat_name = part.replace(' ','')
        n = feat_list.index(feat_name)
        weights[n].append(exp[1])

    #Get average explanation
    weights = np.transpose(weights)
    avg_weight = np.average(np.array(weights), axis = 0)
    abs_weight = [abs(weight) for weight in avg_weight]
    
    bins = pd.cut(abs_weight, 4, retbins = True, duplicates = "drop")
    q1_min = bins[1][-2]
    
    sorted_weight = np.copy(abs_weight)
    sorted_weight.sort()
    
    lime_features = [i for i in range(len(feat_list)) if abs_weight[i] >= q1_min]
    
    return lime_features

#Feature extraction from LINDA-BN
def get_linda_features(instance, cls, scaler, dataset, exp_iter, feat_list):
    label_lst = ["Negative", "Positive"]
    
    feat_pos = []
    lkhoods = []
    
    for i in range(exp_iter):
        [bn, inference, infoBN] = generate_BN_explanations(instance, label_lst, feat_list, "Result", 
                                                                       None, scaler, cls, save_to+"/"+cls_method+"/", dataset, show_in_notebook = False)
        
        #Conduct inference and find posterior for predicted class
        ie = pyAgrum.LazyPropagation(bn)
        result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
        result_proba = result_posterior.loc["Result", label_lst[instance['predictions']]]
        row = instance['original_vector']
        #print(row)

        #Find impact on result for based on each feature
        likelihood = [0]*len(feat_list)

        for j in range(len(feat_list)):
            var_labels = bn.variable(feat_list[j]).labels()
            str_bins = list(var_labels)
            bins = []

            for disc_bin in str_bins:
                disc_bin = disc_bin.strip('"(]')
                cat = [float(val) for val in disc_bin.split(',')]
                bins.append(cat)

            for k in range(len(bins)):
                if k == 0 and row[j] <= bins[k][0]:
                    feat_bin = str_bins[k]
                elif k == len(bins)-1 and row[j] >= bins[k][1]:
                    feat_bin = str_bins[k]
                elif row[j] > bins[k][0] and row[j] <= bins[k][1]:
                    feat_bin = str_bins[k]

            ie = pyAgrum.LazyPropagation(bn)
            ie.setEvidence({feat_list[j]: feat_bin})
            ie.makeInference()
            
            result_posterior = ie.posterior(bn.idFromName("Result")).topandas()
            new_proba = result_posterior.loc["Result", label_lst[instance['predictions']]]
            #print(result_proba, new_proba)
            proba_change = result_proba-new_proba
            likelihood[j] = abs(proba_change)

        lkhoods.append(likelihood)

    #Find average impact   
    bins = pd.cut(np.mean(lkhoods, axis=0), 4, retbins = True, duplicates = "drop")
    q1_min = bins[1][-2]

    #If fixing all features produces the same result for the class,
    #return all features
    if len(set(np.mean(lkhoods, axis=0)))==1:
        feat_pos.extend(range(len(feat_list)))
       # print(lkhoods)
    else:
        feat_pos.extend(list(np.where(np.mean(lkhoods, axis=0) >= q1_min)[0]))

    feat_pos = set(feat_pos)
    #print(feat_pos)
    
    return feat_pos

#Feature extraction from ACV
def get_acv_features(explainer, instance, cls, X_train, exp_iter):
    instance = instance.reshape(1, -1)
    y = cls.predict(instance)
    y_pred = cls.predict(X_train)

    feat_pos = []

    #Get M-SE
    for i in range(exp_iter):
        sdp_importance, sdp_index, size, sdp = explainer.importance_sdp_rf(instance, y, X_train, y_pred)
        feat_pos.extend(sdp_index[0, :size[0]])

    #Find unique features in M-SE
    feat_pos = set(feat_pos)

    return feat_pos

#Choose explanation feature extraction method
def get_explanation_features(explainer, instance, cls, scaler, dataset, X_train, classification, exp_iter, xai_method, feat_list):
    if xai_method == "SHAP":
        feat_pos = get_shap_features(explainer, instance, cls, classification, exp_iter, feat_list)
        
    elif xai_method == "LIME":
        feat_pos = get_lime_features(explainer, instance, cls, classification, exp_iter, feat_list)
        
    elif xai_method == "LINDA":
        feat_pos = get_linda_features(instance, cls, scaler, dataset, exp_iter, feat_list)
    
    elif xai_method == "ACV":
        feat_pos = get_acv_features(explainer, instance, cls, X_train, exp_iter)
        
    explanation_features = [feat_list[i] for i in feat_pos]
    explanation_features = set(explanation_features)
        
    return explanation_features

#Choose true feature extraction method
def get_true_features(cls, instance, cls_method, X_train, feat_list):
    if cls_method == "decision_tree":
        feat_pos = get_tree_features(cls, instance)
        
    elif cls_method == "logit" or cls_method == "lin_reg":
        feat_pos = get_reg_features(cls)
        
    elif cls_method == "nb":
        feat_pos = get_nb_features(cls, instance)
        
    true_features = [feat_list[i] for i in feat_pos]
    true_features = set(true_features)
        
    return true_features

# path to project folder
# please change to your own
PATH = os.getcwd()

dataset = sys.argv[1]
cls_method = sys.argv[2]
xai_method = sys.argv[3]
classification = False if sys.argv[4] == "0" else True

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

with open(dataset_folder+"col_dict.json", "r") as f:
    col_dict = json.load(f)
f.close()

feat_list = [each.replace(' ','_') for each in X_train.columns]

#Get model and scaler
cls = joblib.load(save_to+cls_method+"/cls.joblib")
scaler = joblib.load(save_to+"/scaler.joblib")

#Generate SHAP explainer object
if xai_method == "SHAP":
    if cls_method == "xgboost" or cls_method == "decision_tree":
        explainer = shap.Explainer(cls)
    elif cls_method == "nb":
        if classification:
            explainer = shap.Explainer(cls.predict_proba, X_train)
        else:
            raise TypeError("Cannot run naive bayes with regression")
    else:
        explainer = shap.Explainer(cls, X_train)

#Generate LIME explainer object      
elif xai_method == "LIME":
    if col_dict['discrete'] != None:
        cat_cols = [each.replace(' ','_') for each in col_dict['discrete']]
        col_inds = [feat_list.index(each) for each in cat_cols]
    else:
        col_inds = []
    
    if classification==True:
        class_names=['Negative','Positive']# negative is 0, positive is 1, 0 is left, 1 is right
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = feat_list, 
                                                            class_names=class_names, categorical_features = col_inds,
                                                            discretize_continuous=True)
    else:
        class_names = ['Final Value']
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names = feat_list, 
                                                           class_names=class_names, discretize_continuous=True, 
                                                           categorical_features = col_inds, mode = "regression")

#Generate dataset dictionary for LINDA            
elif xai_method == "LINDA":
    test_dict = generate_local_predictions( test_x, results["Actual"].values, cls, scaler, None )
    explainer = None

#retrieve saved ACV surrogate model
elif xai_method == "ACV":
  explainer = joblib.load(save_to+cls_method+"/acv_explainer.joblib")

compiled_precision = []
compiled_recall = []

#For each instance in testing sample
for i in tqdm(range(len(test_x))):
    instance = test_x[i]

    #Get true features
    true_features = get_true_features(cls, instance, cls_method, X_train.values, feat_list)
    
    if xai_method == "LINDA":
        instance = test_dict[i]
    
    #Get explanation features
    explanation_features = get_explanation_features(explainer, instance, cls, scaler, dataset, X_train.values, classification, exp_iter, xai_method, 
                                                    feat_list)

    #Calculate precision and recall using both sets  
    if len(explanation_features) == 0:
        recall = 0
        precision = 0
    else:
        recall = len(true_features.intersection(explanation_features))/len(true_features)
        precision = len(true_features.intersection(explanation_features))/len(explanation_features)
    
    compiled_precision.append(precision)
    compiled_recall.append(recall)

#Update results  
results[xai_method+" Precision"] = compiled_precision
results[xai_method+" Recall"] = compiled_recall

results.to_csv(os.path.join(save_to, cls_method, "results.csv"), index = False, sep = ";")

print(np.mean(compiled_precision))
print(np.mean(compiled_recall))
