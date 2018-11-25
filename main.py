import read_data
import functions
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
import sklearn
import graphviz as gvz

train_path= ['D://desktop//dataset//block_1.csv','D://desktop//dataset//block_2.csv',
            'D://desktop//dataset//block_3.csv','D://desktop//dataset//block_4.csv',
            'D://desktop//dataset//block_5.csv','D://desktop//dataset//block_6.csv',]
test_path = ['D://desktop//dataset//block_7.csv','D://desktop//dataset//block_8.csv',
             'D://desktop//dataset//block_9.csv',]
vali_path =['D://desktop//dataset//block_10.csv',]

target_label='is_match'

# 读取数据
DataSet=read_data.ReadAData(train_path[0])
for i in range(9):
    ii=i+2
    path1='D://desktop//dataset//block_'
    path2='.csv'
    dataset=read_data.ReadAData(path1+str(ii)+path2)
    DataSet.append(dataset)

DataSett=functions.fromNaNToLable(DataSet)

#从数据中提取训练数据和标签
X=DataSett.drop(target_label,axis=1)
Y=DataSett[target_label]

#将数据分为训练集和测试集
Xtrain,Xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)

#print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#print(DataSett.dtypes)

#============================================生成决策树================================================
clf=tree.DecisionTreeClassifier(criterion='entropy',   #采用信息熵的计算方式
                                splitter='random',     #随机在部分划分点中寻找局部最优
                                max_features=None,     #最大特征数
                                max_depth=100,         #最大深度
                                min_samples_split=5,   #分割所需要的最小样本数
                                min_samples_leaf=5,    #身为叶子所需要的最小样本数
                                min_weight_fraction_leaf=0,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0,
                                #min_impurity_split=0
                                )
clf=clf.fit(Xtrain,ytrain)
#======================================保存决策树=====================================================
dot_data=tree.export_graphviz(clf, out_file=None,
                              feature_names=DataSet.columns[:-1],
                              class_names=['not match','match'],
                              filled=True,rounded=True,
                              special_characters=True)
graph=gvz.Source(dot_data)
graph.render('decision_tree')

#进行预测
predict=clf.predict(Xtest)
#进行评估
functions.evualute(ytest,predict)





