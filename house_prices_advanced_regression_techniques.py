# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:37:59 2019

@author: Justin L Ross
"""

######################################################################
# Load libraries
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

######################################################################
# Declare variables and constants
# Load data, train_df is the training data and test_df is the testing data.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

######################################################################
# Functions and code.

# Prelimnary Analysis

train_df.head()
train_df.describe()
train_df.shape
train_df.keys()

correlations = train_df.corr()
correlations = correlations["SalePrice"].sort_values(ascending=False)
features = correlations.index[1:6]
correlations

training_null = pd.isnull(train_df).sum()
testing_null = pd.isnull(test_df).sum()


# Dealing with null vaues

null = pd.concat([training_null, testing_null], axis=1, keys=["Training", "Testing"])

null_many = null[null.sum(axis=1) > 200]  #a lot of missing values
null_few = null[(null.sum(axis=1) > 0) & (null.sum(axis=1) < 200)]  #not as much missing values

null_many

#you can find these features on the description data file provided

null_has_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]


for i in null_has_meaning:
    training[i].fillna("None", inplace=True)
    testing[i].fillna("None", inplace=True)
    
#### MORE HERE ####  
    
# Random Forest

y = train_df["SalePrice"]

features = ["SaleCondition", "YrSold", "OverallQual", "OverallCond"]
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


######################################################################
# Functions to excude and run.



