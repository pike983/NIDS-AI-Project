# Main script for development and testing
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from DataParser import DataParser
from DataSplitter import DataSplitter

# Load and parse data via panda
# col_names = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','service',
#              'Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_dep','res_bdy_len','Sjit','Djit',
#              'Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
#              'ct_ftp_cmd','ct_srv_src','ct_srv_dest','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat',
#              'Label']
train_set = DataParser("Unified-Train-Set.csv")
test_set = DataParser("Unified-Test-Set.csv")
train_set.label()
test_set.label()
### Code to format and split the data set into test and training sets ###
# database = DataParser("UNSW-NB15-BALANCED-TRAIN.csv")#, col_names)
# database.format()
# database.label()
# db = database.dataset_file
# print("Dataset")
# ds = DataSplitter(db)
# ds.test['attack_cat'] = database.labeler.inverse_transform(ds.test['attack_cat'])
# ds.train['attack_cat'] = database.labeler.inverse_transform(ds.train['attack_cat'])
# pd.DataFrame.to_csv(ds.train, "Unified-Train-Set.csv")
# pd.DataFrame.to_csv(ds.test, "Unified-Test-Set.csv")
### RFE Code ###
# rfe = RFE(estimator=DecisionTreeClassifier(),n_features_to_select=5)
# model = DecisionTreeClassifier()
# # Setup pipelines
# pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# # Ghost in the machine
# Xt, yt = database.get_train()
# X, y = database.get_test()
# print(Xt)
# print(yt)
# pipeline.fit(Xt, yt)
# y_pred = pipeline.predict(X)
# print(y_pred)
print("END")

# ???
# Profit