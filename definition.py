import pandas as pd
import numpy as np
import os
import plotly.express as px


class Data:
    def __init__(self):
        self.df = pd.DataFrame()#pd.read_csv('data/S7.csv')
        self.df_train = self.df.head()
        self.df_test = self.df.head()
        self.train_test_split = 0.7
        self.target = ['label']
        self.features = ['cACCx','cACCy','cACCz','cECG','cEMG','cEDA','cTemp','cResp','wACCx','wACCy','wACCz','wBVP','wEDA','wTEMP']
        self.models = ['DT']
obj_Data = Data()
