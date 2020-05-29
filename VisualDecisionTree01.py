#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# In[2]:


data = pd.read_csv("adult.csv")
data.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
data.head()


# In[4]:


data.shape


# In[6]:


y = data['age']
X = data.drop('age', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
columns = X_train.columns


# In[7]:


# 資料標準化
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)


# In[8]:


from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)


# In[9]:


y_prob = model_tree.predict_proba(X_test)[:,1]
y_pred = np.where(y_prob > 0.5, 1, 0)
model_tree.score(X_test, y_pred)


# In[ ]:


# 視覺化樹圖
data_ = pd.read_csv("adult.csv")
data_feature_name = data_.columns[1:]
#data_target_name = np.unique(data_["class"])
data_target_name = np.dtype([('age',np.int32)]) 
import graphviz
import pydotplus
from sklearn import tree
from IPython.display import Image
import os
os.environ["PATH"] += os.pathsep + r'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_tree = tree.export_graphviz(model_tree,out_file=None,feature_names=data_feature_name,class_names=data_target_name,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_tree)
img = Image(graph.create_png())
graph.write_png("out.png")

