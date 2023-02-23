import pandas as pd
import numpy as np
from sklearn import model_selection as sk_ms

class DataSplitter:    
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.train, self.test = sk_ms.train_test_split(self.dataframe, train_size=0.7, test_size=0.3)
        pass

    def __del__(self):
        pass

    pass
