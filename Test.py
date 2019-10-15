#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import accuracy_score


# ## Load data

# In[2]:


#dir_data = "/Users/romulo/Documents/Dataset/Earthquake Damage/"
dir_data = "Dataset/"
df_x_train = pd.read_csv(dir_data+"train_values.csv",index_col="building_id")
df_y_train = pd.read_csv(dir_data+"train_labels.csv",index_col="building_id")
df_x_test = pd.read_csv(dir_data+"test_values.csv",index_col="building_id")

#df_x_train = df_x_train[:100]
#df_y_train = df_y_train[:100]
#df_x_test = df_x_test[:100]

data_train = df_x_train.merge(df_y_train, how='left', left_index=True, right_index=True)



print("len train:",len(data_train))
print("len train:",len(df_x_test))


# ## Preparate data

# In[14]:


# get y train
y_train = data_train['damage_grade'].values

# remove y of data_train
data_train = data_train.drop('damage_grade', 1)


# In[15]:


# let's put the train data and test data together to make get_dummies and then divide
df_x_all = data_train.append(df_x_test)
print("len all:",len(df_x_all))

# get dummies from cat columns
cat_var = [key for key in dict(df_x_all.dtypes) if dict(df_x_all.dtypes)[key] in ['object'] ]
df_x_all = pd.get_dummies(df_x_all, prefix=cat_var, columns=cat_var)

#divide x_train and x_test
x_train = df_x_all.iloc[:len(data_train)]
x_test = df_x_all.iloc[len(data_train):]


# In[16]:


# clear memory
del df_x_all,df_x_train,df_y_train,data_train,df_x_test


# In[17]:


#get x_dev and y_dev (10% from train)
x_train, x_dev, y_train, y_dev = train_test_split( x_train, y_train, test_size=0.1, random_state=42)


# In[18]:


print("len x_train : %d  len y_train: %d " %(len(x_train),len(y_train)) )
print("len x_dev   : %d  len y_dev  : %d " %(len(x_dev),len(y_dev)) )
print("len x_test  : %d" %(len(x_test)) )


# ## Testing Algorithms from sklearn

# In[19]:


names = ["Nearest Neighbors", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(50),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=8),
    RandomForestClassifier(max_depth=8, n_estimators=1500, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# In[20]:


df_results_sklearn = pd.DataFrame(columns=["algorithm","acc_train","acc_dev","precision_1","recall_1","f1-score_1","precision_2","recall_2","f1-score_2","precision_3","recall_3","f1-score_3"])
for name, clf in zip(names, classifiers):
        print(name)
        clf.fit(x_train, y_train)
        score_train = clf.score(x_train, y_train)
        score_dev = clf.score(x_dev, y_dev)
        print("\tscore_train: ",score_train)
        print("\tscore_dev: ",score_dev)
        print("\n\n")
        
        pred_y_pred = clf.predict(x_dev)
        
        dict_report = classification_report(y_dev, pred_y_pred,output_dict=True)
        precision_1 = dict_report["1"]["precision"]
        precision_2 = dict_report["2"]["precision"]
        precision_3 = dict_report["3"]["precision"]
        
        recall_1 = dict_report["1"]["recall"]
        recall_2 = dict_report["2"]["recall"]
        recall_3 = dict_report["3"]["recall"]
        
        score_1 = dict_report["1"]["f1-score"]
        score_2 = dict_report["2"]["f1-score"]
        score_3 = dict_report["3"]["f1-score"]
        
        df_results_sklearn = df_results_sklearn.append({"algorithm": name,"acc_train":score_train,"acc_dev":score_dev,"precision_1" : precision_1, "recall_1" : recall_1, "f1-score_1" : score_1,"precision_2" : precision_2, "recall_2" : recall_2, "f1-score_2" : score_2,"precision_3" : precision_3, "recall_3" : recall_3, "f1-score_3" : score_3} , ignore_index=True)
        df_results_sklearn.to_csv("results/df_results_sklearn_alg.csv")
        


# ## Using XGBoost

# In[21]:


# My gridsearch (greedy)

df_results_XGBoost = pd.DataFrame(columns=["acc_train","acc_dev","max_depth","n_estimator","precision_1","recall_1","f1-score_1","precision_2","recall_2","f1-score_2","precision_3","recall_3","f1-score_3"])

max_depth = 6
n_estimators =  range(100,2100,100)

for n_estimator in n_estimators:
    print("n_estimators:",n_estimator)

    model = xgb.XGBClassifier(n_estimators=n_estimator,max_depth=max_depth)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_dev_pred = model.predict(x_dev)
    

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_dev = accuracy_score(y_dev, y_dev_pred)
    
    pred_y_pred = clf.predict(x_dev)
        
    dict_report = classification_report(y_dev, pred_y_pred,output_dict=True)
    precision_1 = dict_report["1"]["precision"]
    precision_2 = dict_report["2"]["precision"]
    precision_3 = dict_report["3"]["precision"]

    recall_1 = dict_report["1"]["recall"]
    recall_2 = dict_report["2"]["recall"]
    recall_3 = dict_report["3"]["recall"]

    score_1 = dict_report["1"]["f1-score"]
    score_2 = dict_report["2"]["f1-score"]
    score_3 = dict_report["3"]["f1-score"]
    
    df_results_XGBoost = df_results_XGBoost.append({"acc_train":accuracy_train,"acc_dev":accuracy_dev,"max_depth":max_depth, "n_estimator":n_estimator, "precision_1" : precision_1, "recall_1" : recall_1, "f1-score_1" : score_1,"precision_2" : precision_2, "recall_2" : recall_2, "f1-score_2" : score_2,"precision_3" : precision_3, "recall_3" : recall_3, "f1-score_3" : score_3} , ignore_index=True)
    df_results_XGBoost.to_csv("results/df_results_XGBoost.csv")
    del model
    print("\taccuracy_train",accuracy_train)
    print("\taccuracy_dev",accuracy_dev)
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




