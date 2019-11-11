#!/usr/bin/env python
# coding: utf-8

# In[43]:


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
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample

from featexp import get_trend_stats
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from numpy import sort
import lightgbm as lgb

import time
from feature_selector import FeatureSelector
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[ ]:


get_ipython().system('pip3 install hyperopt')


# ## Load data

# In[2]:


#dir_data = "/Users/romulo/Documents/Dataset/Earthquake Damage/"
dir_data = "Dataset/"
df_x_train = pd.read_csv(dir_data+"train_values.csv",index_col="building_id")
print("df_x_train",df_x_train.shape)
df_y_train = pd.read_csv(dir_data+"train_labels.csv",index_col="building_id")
df_x_test = pd.read_csv(dir_data+"test_values.csv",index_col="building_id")
print("df_x_test",df_x_test.shape)

#df_x_train = df_x_train[:100]
#df_y_train = df_y_train[:100]
#df_x_test = df_x_test[:100]

data_train = df_x_train.merge(df_y_train, how='left', left_index=True, right_index=True)
print("data_train",data_train.shape)


print("len train:",len(data_train))
print("len train:",len(df_x_test))


# ## Preparate data

# In[3]:


# get y train
y_train = data_train['damage_grade'].values

# remove y of data_train
data_train = data_train.drop('damage_grade', 1)


# In[4]:


# let's put the train data and test data together to make get_dummies and then divide
df_x_all = data_train.append(df_x_test)
print("df_x_all:",df_x_all.shape)

# get dummies from cat columns
cat_var = [key for key in dict(df_x_all.dtypes) if dict(df_x_all.dtypes)[key] in ['object'] ]
df_x_all = pd.get_dummies(df_x_all, prefix=cat_var, columns=cat_var)

#divide x_train and x_test
x_train = df_x_all.iloc[:len(data_train)]
x_test = df_x_all.iloc[len(data_train):]

print("x_train:",x_train.shape)     
print("x_test:",x_test.shape)


# In[5]:


x_train.shape


# In[6]:


x_test.shape


# In[7]:


# clear memory
del df_x_all,df_x_train,df_y_train,data_train,df_x_test


# In[8]:


#get x_dev and y_dev (10% from train)
x_train, x_dev, y_train, y_dev = train_test_split( x_train, y_train, test_size=0.1, random_state=44,shuffle=True)


# In[9]:


print("len x_train : %d  len y_train: %d " %(len(x_train),len(y_train)) )
print("len x_dev   : %d  len y_dev  : %d " %(len(x_dev),len(y_dev)) )
print("len x_test  : %d" %(len(x_test)) )


# In[10]:


y_train = np.array([x-1 for x in y_train])
y_dev = np.array([x-1 for x in y_dev])


# In[11]:


#Train Distribution

d = {'y_train': y_train}
df_y_train = pd.DataFrame(d)
print(df_y_train["y_train"].value_counts())
df_y_train["y_train"].value_counts().plot.bar( figsize = (10, 8), rot=45 )


# ## Features

# In[12]:


fs = FeatureSelector(data = x_train, labels = y_train)


# ### Missing Values
# The first method for finding features to remove is straightforward: find features with a fraction of missing values above a specified threshold.

# In[13]:


fs.identify_missing(missing_threshold = 0.1)


# ### Collinear Features 
# Collinear features are features that are highly correlated with one another. In machine learning, these lead to decreased generalization performance on the test set due to high variance and less model interpretability.

# In[14]:


fs.identify_collinear(correlation_threshold = 0.70)
fs.plot_collinear()


# In[15]:


# list of collinear features to remove
collinear_features = fs.ops['collinear']

# dataframe of collinear features
drop_record_collinear = fs.record_collinear.sort_values(by=['corr_value'],ascending=False).head(3)["drop_feature"].values
fs.record_collinear.sort_values(by=['corr_value'],ascending=False).head(3)


# ### Zero Importance Features
# The identify_zero_importance function finds features that have zero importance according to a gradient boosting machine (GBM) learning model.

# In[16]:


# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification', 
                            eval_metric = 'multi_logloss', 
                            n_iterations = 100, early_stopping = True)


# In[17]:


# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
zero_importance_features


# In[18]:


# plot the feature importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 30)


# ### Low Importance Features
# The function identify_low_importance finds the lowest importance features that do not contribute to a specified total importance.Based on the plot of cumulative importance and this information, the gradient boosting machine considers many of the features to be irrelevant for learning

# In[19]:


fs.identify_low_importance(cumulative_importance = 0.99)


# In[20]:


df_feature_importances = fs.feature_importances.sort_values(by=['normalized_importance'],ascending=True)
remove_low_feature_importances = df_feature_importances.head(18)["feature"].values

remove_low_feature_importances


# ### Single Unique Value Features
# The final method is fairly basic: find any columns that have a single unique value. 

# In[21]:


fs.identify_single_unique()
fs.plot_unique()


# ### Removing Features

# In[22]:


remove_feature = list(remove_low_feature_importances)


# In[23]:


x_train = x_train.drop(columns=remove_feature,axis=0)


# In[24]:


x_dev = x_dev.drop(columns=remove_feature,axis=0)


# In[25]:


print(x_train.shape)
print(x_dev.shape)


# ## Testing dev set

# In[26]:


d_train = lgb.Dataset(x_train, label=y_train)
d_val   = lgb.Dataset(x_dev, label=y_dev)


# In[27]:


params = {}
params['learning_rate'] = 0.05
params['boosting_type'] = 'gbdt'
params['objective'] = 'multiclass'
params['metric'] = 'multi_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 25
params['min_data'] = 30
params['max_depth'] = 10
params['num_class'] = 3
clf = lgb.train(params, d_train, 12000,valid_sets=[d_val])


# In[28]:


pred_dev =  clf.predict(x_dev)
pred_dev = np.argmax(pred_dev,axis=1)
accuracy_dev = accuracy_score(y_dev, pred_dev)
accuracy_dev


# In[29]:


#0.7454817543455738
#0.7455201258585626


# In[30]:


print(classification_report(y_dev, pred_dev))


# # Bayesian Optimization

# In[36]:


param_hyperopt= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 1000, 15000, 1000)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 5, 50, 1)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}


# In[38]:


def hyperopt(param_space, X_train, y_train, X_test, y_test, num_eval):
    
    start = time.time()
    
    def objective_function(params):
        clf = lgb.LGBMClassifier(**params)
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function, 
                      param_space, 
                      algo=tpe.suggest, 
                      max_evals=num_eval, 
                      trials=trials,
                      rstate= np.random.RandomState(1))
    loss = [x['result']['loss'] for x in trials.trials]
    
    best_param_values = [x for x in best_param.values()]
    
    if best_param_values[0] == 0:
        boosting_type = 'gbdt'
    else:
        boosting_type= 'dart'
    
    clf_best = lgb.LGBMClassifier(learning_rate=best_param_values[2],
                                  num_leaves=int(best_param_values[5]),
                                  max_depth=int(best_param_values[3]),
                                  n_estimators=int(best_param_values[4]),
                                  boosting_type=boosting_type,
                                  colsample_bytree=best_param_values[1],
                                  reg_lambda=best_param_values[6],
                                 )
                                  
    clf_best.fit(X_train, y_train)
    
    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss)*-1)
    print("Best parameters: ", best_param)
    print("Test Score: ", clf_best.score(X_test, y_test))
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)
    
    return trials


# In[44]:


num_eval = 75
results_hyperopt = hyperopt(param_hyperopt, x_train, y_train, x_dev, y_dev, num_eval)


# ## Testing all for submit

# In[ ]:


x_all = x_train.append(x_dev)


# In[ ]:


y_all = np.concatenate((y_train, y_dev))


# In[ ]:


all_train = lgb.Dataset(x_all, label=y_all)


# In[ ]:


n_est = 14000
clf = lgb.train(params, all_train, n_est)


# In[ ]:





# ## TIME TO PREDICT AND SUBMIT

# In[ ]:





# In[ ]:


#x_test = x_test.drop(columns=remove_feature,axis=0)


# In[ ]:


#predictions = clf.predict(x_test)


# In[ ]:


#predictions = np.argmax(predictions,axis=1)
#predictions = [x+1 for x in predictions]


# In[ ]:


#submission_format = pd.read_csv(dir_data + 'submission_format.csv', index_col='building_id')


# In[ ]:


# my_submission = pd.DataFrame(data=predictions,
#                              columns=submission_format.columns,
#                              index=submission_format.index)


# In[ ]:


#my_submission.to_csv('lgbClassifier_'+str(n_est)+'_all_submission_remove_feature.csv')


# In[ ]:





# In[ ]:




