#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns 
from scipy import stats
from sklearn import metrics
import pickle  
#from prettytable import PrettyTable  

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer
from sklearn.metrics import auc, f1_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict

#get_ipython().run_line_magic('matplotlib', 'inline')
from DataParser import DataParser

# ## Loading data

train = DataParser('./Unified-Train-Set.csv')
test = DataParser('./Unified-Test-Set.csv')
train.label()
test.label()
train_data = train.dataset_file
test_data = test.dataset_file



col_names = train_data.columns
x_names = col_names[1:-2]
y_names = col_names[-2:]
X = train_data[x_names]
y = train_data[y_names]
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)


# ## Correlation

def multi_corr(col1, col2=train_labels['Label'], df=train_features):
    corr = df[[col1, col2]].corr().iloc[0,1]
    log_corr = df[col1].apply(np.log1p).corr(df[col2])
    print("Correlation : {}\nlog_Correlation: {}".format(corr, log_corr))


## Correlation between 2 features

def corr(col1, col2=train_labels['Label'], df=train_features):
    return df[[col1, col2]].corr().iloc[0,1]


# ## Correlation Matrix

method = "pearson"
corr_mat = train_features.corr(method=method)
plt.figure(figsize=(15,15)) 
sns.heatmap(corr_mat, square=True)
plt.show()


# High correlated features
limit = 0.9

columns = corr_mat.columns
for i in range(corr_mat.shape[0]):
    for j in range(i+1, corr_mat.shape[0]):
        if corr_mat.iloc[i, j] >= 0.9:
            print(f"{columns[i]:20s} {columns[j]:20s} {corr_mat.iloc[i, j]}")


limit = 0.9

corr_mat = train_features.corr()
high_corr_features = []
toDrop = []

for i in range(corr_mat.shape[0]):
    for j in range(i+1, corr_mat.shape[0]):
        if corr_mat.iloc[i, j] >= limit:
            high_corr_features.append(corr_mat.columns[i])
            toDrop.append(corr_mat.columns[j])

high_corr_features = list(set(high_corr_features))
toDrop = list(set(toDrop))

print("High Correlated Features: ")
print(high_corr_features)


# Remove duplicates from toDrop if they also exist in high_corr_features
toDrop = list(set(toDrop) - set(high_corr_features))
#toDrop = [x for x in toDrop if x not in ["Label", "attack_cat"]]

print("Features to drop: ")
print(toDrop)


train_features = train_features.drop(columns=toDrop)
#test_features = test_features.drop(columns=toDrop)

print("Train columns:", train_features.shape[1])
print("Test columns:", test_features.shape[1])


X_train = train_features[high_corr_features]
y_train = train_labels
X_test = test_features[high_corr_features]
y_test = test_labels

# Convert categorical features to one-hot encoding
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a decision tree classifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train['Label'])

# Predict the classes for the testing data
y_pred = clf.predict(X_test)
# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test['Label'], y_pred)
error = 100 - (accuracy * 100)
print("Label")
print("Error percentage:", error)

# Calculate the confusion matrix and classification report
cm = confusion_matrix(y_test['Label'], y_pred)
print("Confusion matrix:")
#print(cm)
print("Classification report:")
print(classification_report(y_test['Label'], y_pred))


