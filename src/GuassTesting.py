import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from DataParser import DataParser
from DataSplitter import DataSplitter

print("--- Reading data ---")
train_set = DataParser("Unified-Train-Set.csv")
test_set = DataParser("Unified-Test-Set.csv")
validation_set = DataParser("Unified-Validation-Set.csv")
train_set.label()
test_set.label()
validation_set.label()

label_features = ['sport','dsport','sttl','Sload','Dload','smeansz','Stime','Sintpkt','synack','ct_state_ttl']
cat_features = ['sport','dsport','sbytes','dbytes','sttl','service','smeansz','dmeansz','Stime','Ltime']

print("--- Seperating datasets ---")
X_l_train = train_set.dataset_file[label_features]
X_l_test = test_set.dataset_file[label_features]
y_l_train = train_set.dataset_file['Label']
y_l_test = test_set.dataset_file['Label']

X_c_train = train_set.dataset_file[cat_features]
X_c_test = test_set.dataset_file[cat_features]
y_c_train = train_set.dataset_file['attack_cat']
y_c_test = test_set.dataset_file['attack_cat']
model = GaussianNB()

print("--- Label prediction in progress ---")
y_l_pred = model.fit(X_l_train, y_l_train).predict(X_l_test)

print("--- Label prediction analysis ---")
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_l_test, y_l_pred)*100))
print(metrics.classification_report(y_l_test, y_l_pred))

print("--- Classification in progress ---")
y_c_pred = model.fit(X_c_train, y_c_train).predict(X_c_test)

print("--- Classification prediction analysis ---")
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_c_test, y_c_pred)*100))
print(metrics.classification_report(y_c_test, y_c_pred))