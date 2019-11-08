#!/usr/bin/env python
# coding: utf-8

# In[1]:


import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle
import pandas as pd


# In[2]:


dir_data = "Dataset/"

#dir_data = "Dataset/"
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


# # Preparate data

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


# clear memory
del df_x_all,df_x_train,df_y_train,data_train,df_x_test


# # Train

# In[22]:


dir_ = "/home/dslab/deep_learning/challenge_earthquake/"

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=86400,
    per_run_time_limit=2700,
    #time_left_for_this_task=3600,
    #per_run_time_limitint=1800,
    tmp_folder=dir_+'autosklearn/tmp_folder/',
    output_folder='autosklearn/output_folder/',
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False)


# In[23]:


# fit() changes the data in place, but refit needs the original data. We
# therefore copy the data. In practice, one should reload the data
automl.fit(x_train.copy(), y_train.copy(), dataset_name='quake_data')


# In[24]:


# During fit(), models are fit on individual cross-validation folds. To use
# all available data, we call refit() which trains all models in the
# final ensemble on the whole dataset.
automl.refit(x_train.copy(), y_train.copy())


# In[25]:


predictions = automl.predict(x_test)


# In[26]:


with open('autosklearn/predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)


# In[27]:


submission_format = pd.read_csv(dir_data + 'submission_format.csv', index_col='building_id')


# In[28]:


my_submission = pd.DataFrame(data=predictions,
                             columns=submission_format.columns,
                             index=submission_format.index)


# In[ ]:


my_submission.to_csv('autosklearn/autosklearn_submission.csv')


# In[ ]:





# In[ ]:




