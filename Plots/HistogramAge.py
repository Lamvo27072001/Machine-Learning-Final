# In[0]: Install library
from math import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline #Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean 

# In[1]: Overview of the topic:
# Predict one's salary or make one
# software, tools to predict salary when mik apply
# and mik use the deal salary properly

# In[2]: GET DATASET
'''
    - Data is taken from Kaggle
    Source: https://www.kaggle.com/parulpandey/2020-it-salary-survey-for-eu-region
    - The dataset contains salary survey information of IT personnel in Europe.
    - The dataset used by the group is the survey information of 2020.
    Author: Parul Pandey
    - The last update of the dataset was 2 years ago.
'''
raw_data = pd.read_csv('Raw_DataSet/IT Salary Survey EU  2020.csv')
#%%
data = raw_data["Gender"]
plt.hist(data,color = 'r')


# %%
if 1:
    raw_data.plot(kind="scatter", y="Yearly brutto salary (without bonus and stocks) in thoundsands EUR", x="Age", alpha=0.2)
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()   