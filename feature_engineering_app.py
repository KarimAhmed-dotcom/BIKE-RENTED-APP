#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.base import BaseEstimator,TransformerMixin 
import pandas as pd 
import numpy as np
class feature_engineering_app(BaseEstimator,TransformerMixin) : 
    def __init__(self) : 
        pass
        
    def fit(self,X,y=None) : 
        return self 
    
    def transform(self,X,y=None) : 
        X_copy=X.copy() 
        X_copy=pd.DataFrame(X_copy,columns=['temp','humidity','windspeed','season', 'holiday', 'workingday', 'weather','hour' ,'month_name', 'day_of_week'])
        X_copy['is_rush_hour']=X_copy['hour'].isin([17,18,8,19,16,7,9])
        X_copy['is_weekend']=X_copy['day_of_week'].isin(['Saturday','Sunday'])
        for col in ['temp','humidity','windspeed'] : 
            X_copy[col]=X_copy[col].astype('float64')
        for col in ['season', 'holiday', 'workingday', 'weather','hour' ,'month_name', 'day_of_week','is_rush_hour','is_weekend'] : 
            X_copy[col]=X_copy[col].astype('category')
        return X_copy


# In[ ]:




