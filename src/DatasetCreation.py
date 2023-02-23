import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from DataParser import DataParser
from DataSplitter import DataSplitter

### Code to format and split the data set into test and training sets ###
database = DataParser("UNSW-NB15-BALANCED-TRAIN.csv")
database.format()
database.label()
db = database.dataset_file
ds = DataSplitter(db)
ds.test['attack_cat'] = database.labeler.inverse_transform(ds.test['attack_cat'])
ds.train['attack_cat'] = database.labeler.inverse_transform(ds.train['attack_cat'])
ds.validation['attack_cat'] = database.labeler.inverse_transform(ds.validation['attack_cat'])

pd.DataFrame.to_csv(ds.train, "Unified-Train-Set.csv")
pd.DataFrame.to_csv(ds.test, "Unified-Test-Set.csv")
pd.DataFrame.to_csv(ds.validation, "Unified-Validation-Set.csv")