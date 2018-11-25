import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def fromNaNToLable(data):
    '''
    这个函数的作用是为了将缺省值映射到高维空间
    :param data:即输入的数据
    :return:输出的也是这个数据，不过里面的缺省值全部被换成了标签
    '''
    labels=[ #'id_1','id_2',
            'cmp_fname_c1',
            'cmp_fname_c2',
            'cmp_lname_c2',
            #'cmp_sex',
            'cmp_bd','cmp_bm','cmp_by','cmp_plz'
    ]
    Nanstring='?'
    FALSE='FALSE'
    TRUE='TRUE'
    for label in labels:
        data[label][data[label]==Nanstring]=-1
        data[label][data[label]==FALSE]=0
        data[label][data[label]==TRUE] =1

    #print(data)
    return data


def evualute(ytest,ypredict):
    '''
    这个函数用来进行准确度等的评测
    :param ytest:
    :param ypredict:
    :return:
    '''
    matrix=confusion_matrix(ytest,ypredict)
    report=classification_report(ytest,ypredict)
    # 输出混淆矩阵
    print('==='*20+'评估环接'+'==='*20)
    print('混淆矩阵为：\n{}'.format(matrix))
    print('分类指标报告为：\n')
    print(report)
    all_num=matrix.sum()
    error=(matrix[0,1]+matrix[1,0])/all_num
    print('预测的准确率为：{}'.format(1-error))
    print('\n' * 6)

def plott(data):
    #时间仓促没能完成，实在抱歉！
    #x_min,x_max=data[:]
    return 0




