# Main script for development and testing
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from DataParser import DataParser
from DataSplitter import DataSplitter

# Load and parse data via panda
# col_names = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service',
#              'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_dep','res_bdy_len','Sjit','Djit',
#              'Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
#              'ct_ftp_cmd','ct_srv_src','ct_srv_dest','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat',
#              'Label']
# Setting up testing dataset
train_set = DataParser("Unified-Train-Set.csv")
# test_set = DataParser("Unified-Test-Set.csv")
y_labels = train_set.dataset_file['attack_cat'].unique()
train_set.label()
# test_set.label()
col_names = train_set.dataset_file.columns
x_names = col_names[1:-2]
y_names = col_names[-2:]
X = train_set.dataset_file[x_names]
y = train_set.dataset_file[y_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
y_train_label = y_train['Label']
y_train_cat = y_train.loc[:,'attack_cat']
#y_train_cat = train_set.labeler.fit_transform(y_train_cat.fillna('None'))
y_test_label = y_test['Label']
y_test_cat = y_test['attack_cat']
#y_test_cat = train_set.labeler.fit_transform(y_test_cat.fillna('None'))
# X_train = train_set.dataset_file[x_names]
# y_train = train_set.dataset_file[y_names]
# X_test = test_set.dataset_file[x_names]
# y_test = test_set.dataset_file[y_names]
### Code to format and split the data set into test and training sets ###
# database = DataParser("UNSW-NB15-BALANCED-TRAIN.csv")#, col_names)
# database.format()
# database.label()
# db = database.dataset_file
# ds = DataSplitter(db)
# ds.test['attack_cat'] = database.labeler.inverse_transform(ds.test['attack_cat'])
# ds.train['attack_cat'] = database.labeler.inverse_transform(ds.train['attack_cat'])

# pd.DataFrame.to_csv(ds.train, "Unified-Train-Set.csv")
# pd.DataFrame.to_csv(ds.test, "Unified-Test-Set.csv")
### RFE Code ###
rfe = RFE(estimator=DecisionTreeClassifier(),n_features_to_select=10)
model = DecisionTreeClassifier()
# Setup pipelines
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# Ghost in the machine
pipeline.fit(X_train, y_train_label)
y_pred = pipeline.predict(X_test)
print("---Attack labeling---")
# train_set.labeler.inverse_transform(y_pred)
print(rfe.get_feature_names_out())
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test['Label'], y_pred)*100))
print(metrics.classification_report(y_test['Label'], y_pred))
pipeline.fit(X_train, y_train_cat)
y_pred = pipeline.predict(X_test)
y_test_cat = y_test['attack_cat']
print("---Category labeling---")
print(rfe.get_feature_names_out())
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test_cat, y_pred)*100))
y_pred = train_set.relabel(y_pred)
y_test_cat = train_set.relabel(y_test_cat)
print(metrics.classification_report(y_test_cat, y_pred,labels=y_labels))

print("END")

# ???
# Profit