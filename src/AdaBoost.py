import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
import pickle

from DataParser import DataParser

def runRFEandABC(data, task='Label', pickledModel=None):
    test_set = data
    col_names = test_set.dataset_file.columns
    x_names = col_names[1:-2]
    y_names = col_names[-2:]
    test_features = test_set.dataset_file[x_names]
    test_labels = test_set.dataset_file[y_names]
    test_labels_label = test_labels['Label']
    test_labels_cat = test_labels['attack_cat']
    
    train_set = DataParser("Unified-Train-Set.csv")
    y_labels = train_set.dataset_file['attack_cat'].unique()
    train_set.label()
    col_names = train_set.dataset_file.columns
    x_names = col_names[1:-2]
    y_names = col_names[-2:]
    X = train_set.dataset_file[x_names]
    y = train_set.dataset_file[y_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    y_train_label = y_train['Label']
    y_train_cat = y_train.loc[:,'attack_cat']
    y_test_label = y_test['Label']
    y_test_cat = y_test['attack_cat']

    if pickledModel is None:
        ### RFE Code ###
        rfe = RFE(estimator=DecisionTreeClassifier(random_state=42),n_features_to_select=10)
        model = AdaBoostClassifier(n_estimators=100)
        
        if featureSelection:
            pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    
            # Setting up random search training
            #rf_random = RandomizedSearchCV(estimator = pipeline, param_distributions = random_grid,
            #                               n_iter = 100, cv = 3, verbose=2, random_state=42,
            #                               n_jobs = -1)
        else:
            pipeline = Pipeline(steps=[('m',model)])

    if task == 'Label':
        if pickledModel is None:
            pipeline.fit(X_train, y_train_label)
            if toPickle:
                pickle.dump(pipeline, open('ABC_Label_Model.pkl', 'wb'))
            #rf_random.fit(X_train, y_train_label)

            #print("Best Params for Label")
            #pprint(rf_random.best_params_)
        else:
            pipeline = pickle.load(open(pickledModel, 'rb'))
    
        #y_pred = pipeline.predict(X_test)
        y_pred = pipeline.predict(test_features)
        #print("---Attack labeling---")
        #print(rfe.get_feature_names_out())
        #print("Accuracy: {:.2f}%\n".format(accuracy_score(y_test_label, y_pred)*100))
        #print(classification_report(y_test_label, y_pred, zero_division=0))
        print("Accuracy: {:.2f}%\n".format(accuracy_score(test_labels_label, y_pred)*100))
        print(classification_report(test_labels_label, y_pred, zero_division=0))
    elif task == 'attack_cat':
        if pickledModel is None:
            pipeline.fit(X_train, y_train_cat)
            if toPickle:
                pickle.dump(pipeline, open('ABC_Attack_Model.pkl', 'wb'))
            #rf_random.fit(X_train, y_train_cat)
    
            #print("Best Params for attack_cat")
            #print(rf_random.best_params_)
        else:
            pipeline = pickle.load(open(pickledModel, 'rb'))
    
        #y_pred = pipeline.predict(X_test)
        y_pred = pipeline.predict(test_features)
        #print("---Category labeling---")
        #print(rfe.get_feature_names_out())
        #print("Accuracy: {:.2f}%\n".format(accuracy_score(y_test_cat, y_pred)*100))
        #y_pred = train_set.relabel(y_pred)
        #y_test_cat = train_set.relabel(y_test_cat)
        #print(classification_report(y_test_cat, y_pred,labels=y_labels, zero_division=0))
        print("Accuracy: {:.2f}%\n".format(accuracy_score(test_labels_cat, y_pred)*100))
        y_pred = test_set.relabel(y_pred)
        test_labels_cat = test_set.relabel(test_labels_cat)
        print(classification_report(test_labels_cat, y_pred,labels=y_labels, zero_division=0))
    else:
        print("Invalid task")