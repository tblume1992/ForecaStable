# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:29:25 2020

@author: TXB3Y48
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import scipy

pd.options.mode.chained_assignment = None
sns.set_style("darkgrid")

class ForecaStable:
    
    def __init__(self, error_metric = 'smape', custom_data = False):
        self.error_metric = 'smape'
        self.benchmark_mape = pd.read_excel('m4_benchmark_mape.xlsx')
        self.benchmark_train = pd.read_excel('m4_daily_train.xlsx')
        self.prepared_data = self.prepare_data()
        
        return
        
    def prepare_data(self):
        self.benchmark_train.index = self.benchmark_train['V1']
        self.benchmark_train = self.benchmark_train.drop('V1', axis = 1)
        self.benchmark_mape.index = self.benchmark_mape['id']
        self.benchmark_mape = self.benchmark_mape[[self.error_metric]]
        trimmed = []
        for row in range(len(self.benchmark_train)):
          trimmed.append(self.benchmark_train.iloc[row, :].dropna()[-1095:])
          
        return trimmed
    
    def evaluate(self, eval_func: dict, drop_metric = None):
        if drop_metric:
            self.benchmark_mape = self.benchmark_mape.drop(drop_metric, axis = 1)
        for key, value in eval_func.items():
            func_results = []
            for series in tqdm(self.prepared_data):
              func_results.append(value(series))          
            self.benchmark_mape[key] = func_results
            
        return self.benchmark_mape
    
    def summarize(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_dataset = self.benchmark_mape
        columns = list(self.benchmark_mape)
        columns.remove(self.error_metric)
        for column in columns:       
            scaled_dataset[column] = scaler.fit_transform(np.array(scaled_dataset[column]).reshape(-1,1))  
            sns.regplot(
                        x=self.error_metric,
                        y=column,
                        data=scaled_dataset, 
                        fit_reg=True, 
                        label=column, 
                        robust = False
                        ) 
            
            
            corr = scipy.stats.pearsonr(self.benchmark_mape[column], 
                                        self.benchmark_mape[self.error_metric])
            corr = round(corr[0], 3)
            print(f'{column} Pearson Correlation: {corr}')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    test = {
            'cov': lambda x: scipy.stats.variation(x),
            'std': lambda x: np.std(x),

            }
    fe = ForecaStable()
    bm = fe.evaluate(test)
    fe.summarize()


