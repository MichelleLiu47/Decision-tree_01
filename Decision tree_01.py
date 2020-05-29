#!/usr/bin/env python
# coding: utf-8

# In[1]:


#決策樹的helloworld 使用決策樹對iris數據集進行分類
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
#導入iris 數據集
iris = load_iris()
#初始化DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
# 適配數據
clf = clf.fit(iris.data, iris.target)
#將決策樹以 pdf 格式可視化
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")

