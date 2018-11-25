import Node
import numpy as np
import sklearn.tree as tree
import sklearn.datasets as datasets
import graphviz
#lis=np.ones((1,5))
#print(lis)
#liangzi=Node.Node.entropy(lis)
#print(liangzi)

iris=datasets.load_iris()
tre=tree.DecisionTreeClassifier()
tre=tre.fit(iris.data,iris.target)
dot_data=tree.export_graphviz(tre,out_file=None,feature_names=iris.feature_names, class_names=iris.target_names,
                              filled=True, rounded=True, special_characters= True)
graph=graphviz.Source(dot_data)
graph.render('iris')


