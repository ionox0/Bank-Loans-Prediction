
# coding: utf-8

# # Submission Information:
# 
# ### Team Member 1:
# * UNI:  icj2103
# * Name: Ian Johnson
# 
# ### Team Member 2 [optional]:
# * UNI:  df5890
# * Name: Daniel First

# # Step0 - Import Libraries, Load Data [0 points]
# 
# This is the basic step where you can load the data and create train and test sets for internal validation as per your convinience.

# In[1]:

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

# Features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler, StandardScaler, Imputer, MaxAbsScaler, MinMaxScaler

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LassoCV, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

# Testing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[3]:

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
pd.options.display.max_columns = 999
delim = '\n\n' + '*'*30

holdout = pd.read_csv('data/holdout.csv')
data = pd.read_csv('data/data.csv')

# Map labels to boolean values
data['subscribed'] = data['subscribed'].map(lambda x: 0 if x == 'no' else 1)

# Drop credit_defult as no one in holdout set has 'yes' value

# ********
subscribed = data['subscribed']
data_withtarget=data
# ***********

holdout_ids = holdout['ID']
data = data.drop(["subscribed", "duration", "credit_default"], axis=1)
holdout = holdout.drop(["ID", "duration", "credit_default"], axis=1)


# # Step1 - Exploration and Preparation [10 points]
# 
# In this step, we expect you to look into the data and try to understand it before modeling. This understanding may lead to some basic data preparation steps which are common across the two model sets required.

# In[4]:

categorical_df=data.select_dtypes(include=['object'])
categorical_variables=categorical_df.columns

frames = [categorical_df, subscribed]
categ_and_target = pd.concat(frames,axis=1)


# In[5]:

print("Jobs:")
dict_percentage_job={}
job_categories=categ_and_target.job.unique()

for each_job in job_categories:
    just_that_job_df=categ_and_target[categ_and_target.job==each_job]
    percentage=len(just_that_job_df[just_that_job_df.subscribed==1])/len(just_that_job_df)
    dict_percentage_job[each_job]=percentage
print(dict_percentage_job)

print("Months:")
dict_percentage_month={}
month_categories=categ_and_target.month.unique()

for each_month in month_categories:
    just_that_month_df=categ_and_target[categ_and_target.month==each_month]
    percentage=len(just_that_month_df[just_that_month_df.subscribed==1])/len(just_that_month_df)
    dict_percentage_month[each_month]=percentage
print(dict_percentage_month)


print("Education:")
dict_percentage_edu={}
edu_categories=categ_and_target.education.unique()

for each_edu in edu_categories:
    just_that_edu_df=categ_and_target[categ_and_target.education==each_edu]
    percentage=len(just_that_edu_df[just_that_edu_df.subscribed==1])/len(just_that_edu_df)
    dict_percentage_edu[each_edu]=percentage
print(dict_percentage_edu)


# In[6]:

def month_function(month):
    if month=="dec" or month=="mar" or month=="oct" or month=="sep":
        return 1
    else:
        return 0

def job_function(job):
    if job=="student" or job == "retired":
        return 1
    else:
        return 0

# Note: 0s and 1s switched
def education_function(y):
    if y=="basic.9y" or y=="basic.4y" or y=="basic.6y" or y=="high.school":
        return 0
    else:
        return 1


# In[7]:

def get_counts(x,dict):
    return dict[x]


# In[8]:

def new_feats(data):
    data_withbool=data
    
    data_withbool["monthbool"]=data.month.apply(lambda x:month_function(x))
    data_withbool["jobbool"]=data.job.apply(lambda x:job_function(x))
    data_withbool["educationbool"]=data.education.apply(lambda x:education_function(x))
    
    data_withbool_withcounts=data_withbool
    data_withbool_withcounts["educationcounts"]=data.education.apply(lambda x:get_counts(x,dict_percentage_edu))
    data_withbool_withcounts["jobcounts"]=data.job.apply(lambda x:get_counts(x,dict_percentage_job))
    data_withbool_withcounts["monthcounts"]=data.month.apply(lambda x:get_counts(x,dict_percentage_month))
    
    dict_edu_2={}
    dict_edu_2["basic.4y"]=1
    dict_edu_2["basic.6y"]=2
    dict_edu_2["basic.9y"]=3
    dict_edu_2["high.school"]=4
    dict_edu_2["professional.course"]=5
    dict_edu_2["university.degree"]=6
    dict_edu_2["unknown"]=7
    dict_edu_2["illiterate"]=8

    data_withbool_withcounts["edu_linear"]=data.education.apply(lambda x:get_counts(x,dict_edu_2))
    
    return data_withbool_withcounts


# In[9]:

# Log features
def log_feats(data):
    for c in data.select_dtypes(exclude=['object']).columns:
        data[c + '__log'] = data[c].apply(lambda x: np.log(abs(x) + 0.001))
        
    return data


# In[10]:

# Mean of continuous columns for each category of the categorical columns

def fit_mean_cont_per_cat_group(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    continuous_cols = df.select_dtypes(exclude=['object']).columns
    
    means = {}
    
    for cat in categorical_cols:
        for u in df[cat].unique():
            for con in continuous_cols:
                subset = df[cat] == u
                mean_value = df.loc[subset, con].mean()
                
                if not cat in means:
                    means[cat] = {}
                else:
                    means[cat][u] = mean_value
                
    return means


def transform_mean_cont_per_cat_group(df, means):
    categorical_cols = df.select_dtypes(include=['object']).columns
    continuous_cols = df.select_dtypes(exclude=['object']).columns
    
    for cat in categorical_cols:
        for u in df[cat].unique():
            for con in continuous_cols:
                
                subset = df[cat] == u
                mean_value = means[cat][u]
                df.loc[subset, cat + '__' + con + '__mean'] = mean_value
                
    return df


# In[11]:

# Use mean_cat_con features
print(data.shape)
print(holdout.shape)
means = fit_mean_cont_per_cat_group(data)
data_new_feats_1 = transform_mean_cont_per_cat_group(data, means)
holdout_new_feats_1 = transform_mean_cont_per_cat_group(holdout, means)
print(data_new_feats_1.shape)
print(holdout_new_feats_1.shape)


# In[12]:

# Use Daniel's features
data_new_feats_2 = new_feats(data)
holdout_new_feats_2 = new_feats(holdout)
print(data_new_feats_2.shape)
print(holdout_new_feats_2.shape)


# In[13]:

# Use the log features
data_new_feats_3 = log_feats(data_new_feats_2)
holdout_new_feats_3 = log_feats(holdout_new_feats_2)
print(data_new_feats_3.shape)
print(holdout_new_feats_3.shape)


# In[14]:

# Create dummies
data_dummies = pd.get_dummies(data_new_feats_3)
holdout_dummies = pd.get_dummies(holdout_new_feats_3)
print(data_dummies.shape)
print(holdout_dummies.shape)


# In[15]:

x_train, x_test, y_train, y_test = train_test_split(data_dummies, subscribed, random_state=42)# stratify=subscribed, random_state=42)
x_train.shape


# # Utility GridSearch function

# In[16]:

def grid_search_metrics(pipe, param_grid):
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring='roc_auc')
    grid.fit(x_train, y_train)
    print("Best GS score: {}".format(grid.best_score_))
    print("Best params: {}".format(grid.best_params_))
    score = grid.score(x_test, y_test)
    print("Best test score: {}".format(score))
    print("Overfitting amount: {}".format(grid.best_score_ - score))


# # Step2 - ModelSet1 [35 points]
# 
# In this step, we expect you to perform the following steps relevant to the models you choose for set1:
# 
# * feature engineering
# * validation
# * feature selection
# * final model selection
# 
# You may select up to 5 models in this step for the purpose of final ensemble. Any classification algorithm covered in class apart from tree-based models can be tested here.

# ### Logistic Regression - some tuning done

# In[17]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif)),
    ("polys", PolynomialFeatures()),
    ("scaling", MinMaxScaler()),
    ("selection_2", SelectKBest(score_func=f_classif)),
    ("model", LogisticRegression())
])

param_grid = {
    'model__C': [.5], #[1, 0.5, .1, .001]
}

grid_search_metrics(pipe, param_grid)


# # Step3 - ModelSet2 [35 points]
# 
# In this step, we expect you to perform the following steps relevant to the models you choose for set2:
# 
# * feature engineering
# * validation
# * feature selection
# * final model selection
# 
# You may select up to 5 models in this step for the purpose of final ensemble. We encourage you to try decition tree, random forest and gradient boosted tree methods here and pick the one which you think works best.

# ### Random Forest Classifier

# In[18]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif)),
    ("polys", PolynomialFeatures()),
    ("scaling", StandardScaler()),
    ("selection_2", SelectKBest(score_func=f_classif)),
    ("model", RandomForestClassifier())
])

param_grid = {
    'selection_1__k': [55], #[45, 50, 55],
    'selection_2__k': [600], #[600, 700, 800],
    
    "model__max_depth": [3], #[3, None],
    "model__max_features": [14], #[13, 14, 15],
    "model__min_samples_split": [2], #[2, 3, 4],
    "model__min_samples_leaf": [10], #[7, 10, 15],
    "model__bootstrap": [True], #[True, False],
    "model__criterion": ['entropy'], #["gini", "entropy"]
}

grid_search_metrics(pipe, param_grid)


# ### Gradient Boosting Classifier 

# In[19]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif)),
    ("polys", PolynomialFeatures()),
    ("scaling", StandardScaler()),
    ("selection_2", SelectKBest(score_func=f_classif)),
    ("model", GradientBoostingClassifier())
])

param_grid = {
    'selection_1__k': [50],
    'selection_2__k': [700],
    
    'model__max_features': ['sqrt'],
    'model__learning_rate': [.1],
    'model__max_depth': [6], # [5, 6, 7],
    'model__min_samples_leaf': [40], # [39, 40, 41],
    'model__min_samples_split': [600], #[590, 600, 610],
    'model__random_state': [10],
    'model__subsample': [0.8], #[0.75, 0.8, 0.85],
}

grid_search_metrics(pipe, param_grid)


# ### Multi-Layer Perceptron Classifier

# In[20]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif)),
    ("polys", PolynomialFeatures()),
    ("scaling", StandardScaler()),
    ("selection_2", SelectKBest(score_func=f_classif)),
    ("model", MLPClassifier())
])

param_grid = {
    'selection_1__k': [50],
    'selection_2__k': [700],
    
    'model__alpha': [0.0001], #, 0.001],
    'model__hidden_layer_sizes': [(10, 10, 10)], #[(5, 5, 5), (10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40)]
}

grid_search_metrics(pipe, param_grid)


# # Step4 - Ensemble [20 points + 10 Bonus points]
# 
# In this step, we expect you to use the models created before and create new predictions. You should definitely try poor man's stacking but we encourage you to think of different ensemble techniques as well. We will judge your creativity and improvement in model performance using ensemble models and you can potentially earn 10 bonus points here.

# In[21]:

# svc = LinearSVC(C=0.5)
# clf1 = CalibratedClassifierCV(svc, method='sigmoid')

lr = LogisticRegression(C=0.5)
clf1 = CalibratedClassifierCV(lr, method='sigmoid')

rf1 = RandomForestClassifier(
    max_features = 10,
    criterion = 'gini',
    max_depth = 3,
    min_samples_leaf = 10,
    min_samples_split = 3,
    bootstrap = True,
)
clf2 = CalibratedClassifierCV(rf1, method='sigmoid')

gb1 = GradientBoostingClassifier(
    max_features='sqrt',
    learning_rate=0.1,
    max_depth=6,
    min_samples_leaf=40,
    min_samples_split=600,
    random_state=10,
    subsample=0.8
)
clf3 = CalibratedClassifierCV(gb1, method='sigmoid')

clf4 = MLPClassifier(alpha = 0.0001, hidden_layer_sizes = (20, 20, 20))


eclf1 = VotingClassifier(voting='soft', estimators=[
    ('one', clf1),
    ('two', clf2),
    ('three', clf3),
    ('four', clf4),
])


# In[22]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif)),
    ("polys", PolynomialFeatures()),
    ("scaling", StandardScaler()),
    ("selection_2", SelectKBest(score_func=f_classif)),
    ("model", eclf1)
])


# In[23]:

param_grid = {
    'selection_1__k': [70],
    'selection_2__k': [900],
    'model__voting': ['soft'],
}

grid_search_metrics(pipe, param_grid)


# # Stacking

# In[24]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif, k=50)),
    ("polys", PolynomialFeatures()),
    ("scaling", StandardScaler()),
    ("selection_2", SelectKBest(score_func=f_classif, k=700)),
    ("model", eclf1)
])


# In[ ]:

from sklearn.preprocessing import FunctionTransformer

reshaper = FunctionTransformer(lambda X_: np.rollaxis(X_, 1).reshape(-1, 8)[:, 1::2], validate=False)
stacking = Pipeline([
    ('first_stage', eclf1),
    ('reshaper', reshaper),
    ('second_stage', LogisticRegression(C=100))
])

stacking.fit(x_train, y_train)
train_score = stacking.score(x_train, y_train)
test_score = stacking.score(x_test, y_test)
test_preds = stacking.predict_proba(x_test)


# In[ ]:

final_score = roc_auc_score(y_test, test_preds[:,1])
print(final_score)


# In[ ]:

assert final_score > 0.79


# # Holdout Predictions

# In[115]:

pipe = Pipeline([
    ("variance", VarianceThreshold()),
    ("selection_1", SelectKBest(score_func=f_classif, k=50)),
    ("polys", PolynomialFeatures()),
    ("scaling", StandardScaler()),
    ("selection_2", SelectKBest(score_func=f_classif, k=700)),
    ("model", eclf1)
])

from sklearn.preprocessing import FunctionTransformer

reshaper = FunctionTransformer(lambda X_: np.rollaxis(X_, 1).reshape(-1, 8)[:, 1::2], validate=False)
stacking = Pipeline([
    ('first_stage', eclf1),
    ('reshaper', reshaper),
    ('second_stage', LogisticRegression(C=100))
])

stacking.fit(x_train, y_train)
train_score = stacking.score(x_train, y_train)
test_score = stacking.score(x_test, y_test)
test_preds = stacking.predict_proba(x_test)
print(train_score, test_score)


# In[116]:

preds = stacking.fit(data_dummies, subscribed).predict_proba(holdout_dummies)


# In[117]:

preds_subscribed = pd.DataFrame(preds[:,1], columns=['subscribed'])
submission = pd.concat([holdout_ids, preds_subscribed], axis=1)
submission.to_csv(path_or_buf='preds.csv', sep=',', index=False)


# In[ ]:



