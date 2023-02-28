# Class that parses data file
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataParser:

    def __init__(self, file_path):#,col_names):
        self.labeler = LabelEncoder()
        # self.col_names = col_names
        self.dataset_file = pd.read_csv(file_path)#, names=self.col_names, usecols=self.col_names)
        self.dataset_file['attack_cat'] = self.dataset_file.loc[:,'attack_cat'].str.strip().fillna('None')
        pass

    def __del__(self):
        pass

    def feature_select(self):
        label_features = ['sport','dsport','sbytes','sttl','dtcpb','dmeansz','Ltime','Sintpkt','ct_state_ttl','ct_srv_src']
        cat_features = ['sport','dsport','sbytes','dbytes','sttl','service','smeansz','dmeansz','Stime','Sintpkt']

        return self.dataset_file[label_features], self.dataset_file[cat_features]
        pass

    def label_select(self):
        return self.dataset_file['Label'], self.dataset_file['attack_cat']
    
    def format(self):
         # Convert categories to useable data
        self.dataset_file['proto'] = pd.factorize(self.dataset_file['proto'])[0]
        self.dataset_file['state'] = pd.factorize(self.dataset_file['state'])[0]
        self.dataset_file['service'] = pd.factorize(self.dataset_file['service'])[0]
        # Fill in blank integer values with zero where appropriate
        self.dataset_file['ct_ftp_cmd'] = pd.to_numeric(self.dataset_file['ct_ftp_cmd'],errors='coerce')
        self.dataset_file['ct_ftp_cmd'] = self.dataset_file['ct_ftp_cmd'].fillna(0)
        self.dataset_file['ct_flw_http_mthd'] = pd.to_numeric(self.dataset_file['ct_flw_http_mthd'],errors='coerce')
        self.dataset_file['ct_flw_http_mthd'] = self.dataset_file['ct_flw_http_mthd'].fillna(0)
        # Drop rows and columns as appropriate
        self.dataset_file['sport'] = pd.to_numeric(self.dataset_file['sport'],errors='coerce')
        #self.dataset_file['sport'].dropna(axis=0,inplace=True)
        #self.dataset_file['sport'] = self.dataset_file['sport'].fillna(0)
        self.dataset_file['dsport'] = pd.to_numeric(self.dataset_file['dsport'],errors='coerce')
        #self.dataset_file['dsport].dropna(axis=0,inplace=True)
        self.dataset_file.dropna(subset=['sport','dsport'],inplace=True)
        if 'is_ftp_login' in self.dataset_file.columns:
            self.dataset_file.drop('is_ftp_login',axis=1,inplace=True)
        if 'srcip' in self.dataset_file.columns:
            self.dataset_file.drop('srcip',axis=1,inplace=True)
        if 'dstip' in self.dataset_file.columns:
            self.dataset_file.drop('dstip',axis=1,inplace=True)
        #self.dataset_file.drop(['is_ftp_login', 'srcip', 'dstip'], axis=1, inplace=True)
        pass

    def label(self):
        self.dataset_file['attack_cat'] = self.labeler.fit_transform(self.dataset_file['attack_cat'])

    def relabel(self, ypred):
        return self.labeler.inverse_transform(ypred)