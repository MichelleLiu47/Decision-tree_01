#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import load_iris 
iris = load_iris() 
# 模型(也可用單個決策樹) 
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators=10) 
# 訓練 
model.fit(iris.data, iris.target) 
# 提取一個決策樹 
estimator = model.estimators_[5] 
from sklearn.tree import export_graphviz 
# 導出為dot 文件 
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names, 
                class_names = iris.target_names, 
                rounded = True, proportion = False, 
                precision = 2, filled = True) 
# 用系統命令轉為PNG文件(需要 Graphviz)
from subprocess import call 
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600']) 
# 在jupyter notebook中展示 
from IPython.display import Image 
Image(filename = 'tree.png')


# In[ ]:




