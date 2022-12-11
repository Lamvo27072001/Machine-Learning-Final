
'''
FINAL PROJECT MACHINE LEARNING
GROUP 11 : HUYNH NGUYEN KHANG - 19110144
           PHAM VO HONG LAM - 19110154
           HUYNH CONG DAT - 19110114
            
TOPIC: FORECASTING SALARY OF IT HUMAN RESOURCES IN EUROPE IN 2020

'''
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
import json
# In[1]: Topic overview:
# Predict one's salary or make one
# software, tools to predict salary when applying
# and use the deal salary appropriately.
# Predict a person's salary with their qualifications (Programming language, Framework, Cloud, ...)

# In[2]: GET DATASET AND LOAD DATA
'''
    - Data is taken from Kaggle
    Source: https://www.kaggle.com/parulpandey/2020-it-salary-survey-for-eu-region
    - The dataset contains salary survey information of IT personnel in Europe.
    - The dataset used by the group is the survey information of 2020.
    Author: Parul Pandey
    - The last update of the dataset was 2 years ago.
'''
raw_data = pd.read_csv('Raw_DataSet/IT Salary Survey EU  2020.csv')

# In[3]: DISCOVER THE DATA TO GAIN INSIGHTS
# 3.1 Quick view of the data
print('\n____________________________________ Dataset info ____________________________________')
print(raw_data.info())              
print('\n____________________________________ Some first data examples ____________________________________')
print(raw_data.head(6)) 
print('\n____________________________________ Counts on a feature ____________________________________')
print(raw_data['Your main technology / programming language'].value_counts()) 
print('\n____________________________________ Statistics of numeric features ____________________________________')
print(raw_data.describe())    
print('\n____________________________________ Get specific rows and cols ____________________________________')     
print(raw_data.iloc[[0,5,20], [7, 8]] ) # Refer using column ID
#%% PLOT FEATURES
# 3.2. Scatter plot between age and salary of IT workers in Europe in 2020
if 1:
    raw_data.plot(kind="scatter", y="Yearly brutto salary (without bonus and stocks) in thoundsands EUR", x="Age", alpha=0.2)
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()      

# 3.3. Histogram of gender statistics of IT human resources in Europe in 2020:
if 1:
    data = raw_data["Gender"]
    plt.hist(data,color = 'r')
# 3.4. Histogram of the age of IT workers in Europe in 2020:
if 1:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["Age"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.show()

# 3.5 Histogram of salaries of IT workers (Not including bonuses) in Europe in 2020:
if 1:
    data = raw_data["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"]
    plt.hist(data,color = 'r')

#%% 3.6 Compute correlations between features
corr_matrix = raw_data.corr()
print(corr_matrix) 
#%% Filter the data
# Remove samples with too many nulls and outliers
# Re-normalize the data (Fix typos, font errors and uniform capitalization)
#%% 3.7 Add new features
#Function for classifying data:
#%%3.7.1 Implement additional data columns about programming languages
# and add that column and programming language data:
# Install json library to read json files as data dictionaries

# Declare a data variable containing data in the form of dictionaries read from the Json file
data=None
file_path='Json_Files/ListofPL.json'
# Open file Json
with open(file_path) as json_file:
    data = json.load(json_file)
# Check that the letter belongs to the alphabet
def CheckAlphabet(text_check):
    if((text_check >= 'a' and text_check <= 'z') or (text_check >= 'A' and text_check <= 'Z')):
        return True
    else:
        return False
# Check that the programming language is C
def CLanguagesChecking(text_raw_data,key="c"):
    if(text_raw_data.find("Embedded C")!=-1):
        return True
    index=text_raw_data.find(key)
    if(index==-1):
        return False
    index_increased=index+1
    index_decreased=index-1
    try:
        if(text_raw_data[index_increased]==","or text_raw_data[index_increased]=="/"):
            if(text_raw_data[index_decreased]=="-"):
                return False
            return True
        elif text_raw_data[index_increased]=="#" or text_raw_data[index_increased]=="+":
            return False
        elif(text_raw_data[index_decreased]=="/" and (text_raw_data[index_increased]!="#" and text_raw_data[index_increased]!="+" )):
            if((CheckAlphabet(text_raw_data[index_increased]) and CheckAlphabet(text_raw_data[index_decreased]))==False):
                return True
            return False
        elif(CheckAlphabet(text_raw_data[index_increased])):
            return False
        elif((CheckAlphabet(text_raw_data[index_increased]) and CheckAlphabet(text_raw_data[index_decreased]))==False):
            return True
    except:
        if((text_raw_data[index]=="c"or text_raw_data[index]=="C")and text_raw_data[index-1]!="-"):
            return True
        else:
            return False
    return False
# Data classification
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        if(result!=''):
            return result
        index=text_raw_data.find(key)
        if(index!=-1):
            if(key=="C" or key=="c" or key=="R"):
                if(text_raw_data=="C" or text_raw_data=="c"):
                    result+=data[key]+"; "
                elif(key=="R"):
                    if text_raw_data=="R":
                        result+=data[key]+"; "
                    elif CheckAlphabet(text_raw_data[index+1])==False:
                        result+=data[key]+"; "
                elif(CLanguagesChecking(text_raw_data,"c")and key=="c"):
                    result+=data[key]+"; "
                elif(CLanguagesChecking(text_raw_data,"C")and key=="C"):
                    result+=data[key]+"; "
            else:
                if(key=="go"):
                    index_find_Django=text_raw_data.find("Django")
                    if(index!=-1):
                        continue
                if(key=="Java" or key=="java"):
                    try:
                        if(text_raw_data[index+4]=="s"or text_raw_data[index+4]=="S"):
                            continue
                    except:
                        result+=data[key]+"; "
                        continue
                if(key=="Js"and text_raw_data=="Js, reactJS "):
                    continue
                result+=data[key]+"; "
    return result
#%% execute
n=raw_data.index
num=len(n)
# Variables contain programming language data after being processed
# and prepare to insert into dataframe
list_Programming_Languages=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Programming_Languages.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Programming_Languages.append(p)
# Data after being processed and inserted into the dataframe
raw_data["Programming languages"]=list_Programming_Languages

#%%3.7.2 Add data columns and categorize frameworks or libraries

data=None
file_path='Json_Files/ListFM.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        if(result!=''):
            return result
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%% Execute
list_Frameworks_Libraries=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Frameworks_Libraries.append('No info; ')
    else:
        p=ClassifyData(data,s)
        list_Frameworks_Libraries.append(p)
raw_data["Frameworks / Libs"]=list_Frameworks_Libraries
#%%3.7.3 Add data columns and categorize frameworks or libraries

data=None
file_path='Json_Files/ListDB.json'
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%% Execute
list_Databases=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Databases.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Databases.append(p)
raw_data["Databases"]=list_Databases
#%%3.7.4 Add data columns and categorize data about tools and design models

data=None
file_path='Json_Files/ListDesign.json'
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%% Execute
list_Design=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Design.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Design.append(p)
raw_data["Design"]=list_Design
#%%3.7.5 Add data and categorize data about Cloud

data=None
file_path='Json_Files/ListCloud.json'
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%% Execute
list_CLouds=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_CLouds.append('No info')
    else:
        p=ClassifyData(data,s)
        list_CLouds.append(p)
raw_data["Clouds"]=list_CLouds
#%%3.7.6 Platforms

data=None
file_path='Json_Files/ListPlatform.json'
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            if(key=="Linux Kernel"):
                continue
            result+=data[key]+"; "
    return result
#%% Execute
list_Platforms=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_Platforms.append('No info')
    else:
        p=ClassifyData(data,s)
        list_Platforms.append(p)
raw_data["Platform"]=list_Platforms
#%%3.7.7 Add DevOps tools data column data

data=None
file_path='Json_Files/ListDevOps-Tools.json'
with open(file_path) as json_file:
    data = json.load(json_file)

def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]+"; "
    return result
#%% Execute
list_DevOps_Tools=[]
for x in range(num):
    s=raw_data["Your main technology / programming language"][x]
    if(isinstance(s, float)):
        list_DevOps_Tools.append('No info')
    else:
        p=ClassifyData(data,s)
        list_DevOps_Tools.append(p)
raw_data["DevOps tools"]=list_DevOps_Tools
#%%3.7.8. Adjust data total years of experience

data=None
file_path='Json_Files/YearOfExperience.json'
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%% Execute
list_Year_Experience=[]
for x in range(num):
    try:
        s=raw_data["Total years of experience"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Year_Experience.append(s)
            continue
        else:
            list_Year_Experience.append(result)
    except:
        list_Year_Experience.append(raw_data["Total years of experience"][x])
        continue
raw_data["Total years of experience"]=list_Year_Experience
#%%3.7.9. Adjusting contract types data

data=None
file_path='Json_Files/TypeContract.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%% Execute
list_Contract=[]
for x in range(num):
    try:
        s=raw_data["Contract duration"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Contract.append(s)
            continue
        else:
            list_Contract.append(result)
    except:
        list_Contract.append(raw_data["Contract duration"][x])
        continue
raw_data["Contract duration"]=list_Contract
#%%3.7.10. Adjust company size data

data=None
file_path='Json_Files/CompanySize.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%% Execute
list_Company_Size=[]
for x in range(num):
    try:
        s=raw_data["Company size"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Company_Size.append(s)
            continue
        else:
            list_Company_Size.append(result)
    except:
        list_Company_Size.append(raw_data["Company size"][x])
        continue
raw_data["Company size"]=list_Company_Size
#%%3.7.11. Adjust company type data

data=None
file_path='Json_Files/Company_type.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%% Execute
list_Company_Type=[]
for x in range(num):
    try:
        s=raw_data["Company type"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Company_Type.append(s)
            continue
        else:
            list_Company_Type.append(result)
    except:
        list_Company_Type.append(raw_data["Company type"][x])
        continue
raw_data["Company type"]=list_Company_Type
#%%3.7.12. Adjustment of support and subsidy data in 2020

data=None
file_path='Json_Files/Additional_monetary_support.json'
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%% Execute
list_Additional_Support=[]
for x in range(num):
    try:
        s=raw_data["Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Additional_Support.append(s)
            continue
        else:
            list_Additional_Support.append(result)
    except:
        list_Additional_Support.append(raw_data["Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR"][x])
        continue
raw_data["Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR"]=list_Additional_Support
#%%3.7.13. Adjusting the data of job positions of IT human resources in Europe in 2020

data=None
file_path='Json_Files/ListPositions.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    if(text_raw_data=="Analyst" or text_raw_data=="Marketing Analyst" or text_raw_data=="Product Analyst"):
        result='Analytics engineer'
        return result
    elif(text_raw_data=="Consultant" or text_raw_data=="Application Consultant"
     or text_raw_data=="BI Consultant" or text_raw_data=="BI IT Consultant"
      or text_raw_data=="ERP Consultant" or text_raw_data=="SAP BW Senior Consultant" 
      or text_raw_data=="SAP Consultant"):
        result='Consultant'
        return result
    elif(text_raw_data=="Architect" or text_raw_data=="Data Architect"):
        result='Architect'
        return result
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    return result
#%% Execute
list_Positions=[]
for x in range(num):
    try:
        s=raw_data["Position "][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Positions.append(s)
            continue
        else:
            list_Positions.append(result)
    except:
        list_Positions.append(raw_data["Position "][x])
        continue
raw_data["Position "]=list_Positions
#%%3.7.14. Adjustment of data employment

data=None
file_path='Json_Files/ListEmployment.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    if(result==''):
        return text_raw_data
    return result
#%% Execute
list_Employment=[]
for x in range(num):
    try:
        s=raw_data["Employment status"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Employment.append(s)
            continue
        else:
            list_Employment.append(result)
    except:
        list_Employment.append(raw_data["Employment status"][x])
        continue
raw_data["Employment status"]=list_Employment
#%%3.7.15. Adjust Seniority Level data:

data=None
file_path='Json_Files/ListSeniorityLevel.json'
# Opening JSON file
with open(file_path) as json_file:
    data = json.load(json_file)
def ClassifyData(data,text_raw_data):
    result=''
    keys=data.keys()
    for key in keys:
        index=text_raw_data.find(key)
        if(index!=-1):
            result+=data[key]
    if(result==''):
        return text_raw_data
    return result
#%% Execute
list_Seniority_Level=[]
for x in range(num):
    try:
        s=raw_data["Seniority level"][x]
        result=ClassifyData(data,s)
        if(result==''):
            list_Seniority_Level.append(s)
            continue
        else:
            list_Seniority_Level.append(result)
    except:
        list_Seniority_Level.append(raw_data["Seniority level"][x])
        continue
raw_data["Seniority level"]=list_Seniority_Level
# In[04]: PREPARE THE DATA
# 4.1 Remove unused columns
raw_data.drop(columns = ["Timestamp", "Age", "Gender", "City", 
                         "Years of experience in Germany", "Other technologies/programming languages you use often",
                         "Yearly bonus + stocks in EUR", "Annual brutto salary (without bonus and stocks) one year ago. Only answer if staying in the same country",
                         "Annual bonus+stocks one year ago. Only answer if staying in same country","Number of vacation days",
                         "Main language at work","Have you lost your job due to the coronavirus outbreak?",
                         "Have you been forced to have a shorter working week (Kurzarbeit)? If yes, how many hours per week",
                         "Your main technology / programming language"], inplace=True) 
#%% Save the dataset through the preprocessing step
raw_data.to_csv(r'.\DataSet_Filtered\export_dataset.csv', index = False, header=True)
#%% 4.2 Split the above dataset into train set and test set
split = True
if split: # Stratified sampling
    # Create new attribute "Salary_About"
    raw_data["Salary_About"] = pd.cut(raw_data["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"],
                                    bins=[0, 100, 200, 300, 400, np.inf],
                                    labels=[100,200,300,400,500]) # use numeric labels to plot histogram
    
    # Create training set and test set; Practice test to get to the last step
    from sklearn.model_selection import StratifiedShuffleSplit  
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # n_splits: no. of re-shuffling & splitting = no. of train-test sets 
                                                                                  # (if you want to run the algorithm n_splits times with different train-test set)
    for train_index, test_index in splitter.split(raw_data, raw_data["Salary_About"]): # Feature "KHOẢNG GIÁ" must NOT contain NaN
        train_set = raw_data.loc[train_index]
        test_set = raw_data.loc[test_index]      
    
    # Plot sampling histogram in the data set
    if 1:
        raw_data["Salary_About"].hist(bins=6, figsize=(5,5)); #plt.show();
        test_set["Salary_About"].hist(bins=6, figsize=(5,5)); plt.show()

    #Remove the salary_about attribute
    for _set_ in (train_set, test_set):
        _set_.drop(columns="Salary_About", inplace=True) 
    print(train_set.info())
    print(test_set.info())
print('\n____________________________________ Split training an test set ____________________________________')     
print(len(train_set), "train +", len(test_set), "test examples")
#print(train_set.head(4))

#%% 4.3 Separate the label column in the train set and the test set and do not process the label column
# In this project, the label - problem output - is:
# salary column in thousands of EUR.
train_set_labels = train_set["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"].copy()
train_set = train_set.drop(columns = "Yearly brutto salary (without bonus and stocks) in thoundsands EUR") 
test_set_labels = test_set["Yearly brutto salary (without bonus and stocks) in thoundsands EUR"].copy()
test_set = test_set.drop(columns = "Yearly brutto salary (without bonus and stocks) in thoundsands EUR") 

#%% 4.4 Define Pipelines to process data
# INFO: Pipeline is a sequence of transformers (see Geron 2019, page 73). For step-by-step manipulation, see Details_toPipeline.py 

# 4.4.1 Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         
# Save a list file containing columns of numeric data:
num_feat_names = ['Total years of experience', 
'Have you received additional monetary support from your employer due to Work From Home? If yes, how much in 2020 in EUR']
# Save the list file containing the column of literal data (Category):
cat_feat_names = ['Position ', 'Seniority level', 'Employment status',
'Contract duration','Company size','Company type','Programming languages','Frameworks / Libs',
'Databases','Design','Clouds','Platform','DevOps tools']

# 4.4.2 Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    #Fill in the word "No info" when detecting an empty text field
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "No info", copy=True)),
     # Fill in the word "No info" when detecting an empty text field
    ('cat_encoder', OneHotEncoder())
    ])    

# 4.4.4 Pipeline for numerical features:
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))
    ])  
# feature scaling transforms the range of attributes equally, such as 0-1 or -1 - 1 . instead of 1-500 
# Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# 4.5 Run the pipeline to process training data        
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________________________________ Processed feature values ____________________________________')
print(processed_train_set_val[[0, 1, 2],:].toarray())
print(processed_train_set_val.shape)

# In[5]: TRAINING MODELS AND VALIDATION TEST  
'''
    In this step, run the models, measure the accuracy and deviation in each model
    Run validation to ensure objective model selection
    After selecting the model, proceed to the fine-tune step
    In this step the models are made:
    + Linear Regression
    + Decision tree
    + Random Forest
    + Polinomial
'''
#%%Store models to files, to compare latter
import joblib # new lib
def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'saved_objects/' + model_name + '_model.pkl')
def load_model(model_name):
    model = joblib.load('saved_objects/' + model_name + '_model.pkl')
    return model

# 5.1 LinearRegression model
# 5.1.1 Training: learn a linear regression hypothesis using training data 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
print('\n____________________________________ LinearRegression ____________________________________')
print('Learned parameters: ', model.coef_)

# 5.1.2 Compute R2 score and root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
        
#%% 5.1.3 Predict labels for some training instances
print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))
store_model(model)



#%% 5.2 DecisionTreeRegressor model
# Training
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ DecisionTreeRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)
# Predict labels for some training instances
print("Input data: \n", train_set.iloc[0:9])
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.3 RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5, random_state=42) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________________________________ RandomForestRegressor ____________________________________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
print("Predictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.4 Polinomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) 
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# 5.4.2 Compute R2 score and root mean squared error
print('\n____________________________________ Polinomial regression ____________________________________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('R2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("Predictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.5 Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score
print('\n____________________________________ K-fold cross validation ____________________________________')

run_evaluation = 1
if run_evaluation:
    # Model LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))

    # Model DecisionTreeRegressor
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))

    # Model RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    
    # Model Polinomial regression
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')


# In[6]: FINE-TUNE MODELS 
# INFO: Find out hyperparams
print('\n____________________________________ Fine-tune models ____________________________________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    print('Best estimator: ', grid_search.best_estimator_)  
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 1
if method == 1:
    from sklearn.model_selection import GridSearchCV
    
    run_new_search = 1      
    if run_new_search:
        # Fine-tune RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            # then try 6 (2×3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]} ]
            # Train across 5 folds, hence a total of (12+6)*5=90 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 


# In[7]: ANALYZE AND TEST YOUR SOLUTION
# NOTE: solution is the best model from the previous steps. 

# 7.1 Pick the best model - the SOLUTION
# Pick Random forest
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_
# Pick Linear regression
#best_model = joblib.load('saved_objects/LinearRegression_model.pkl')

print('\n____________________________________ ANALYZE AND TEST YOUR SOLUTION ____________________________________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUION")   

# 7.2 Analyse the SOLUTION to get more insights about the data

# 7.3 Run on test data
processed_test_set = full_pipeline.transform(test_set)  
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('Performance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("Test data:", test_set.iloc[0:9])
print("Predictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')


'''
Comment on test results:
The results are not good.
1. Remove unusual samples
2. Add data
'''

# %%
