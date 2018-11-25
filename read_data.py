import pandas as pd
import numpy as np
import functions



path='D://desktop//block_1.csv' #存储所有需要进行读取的文件列表

def ReadAData(path, drop_list=[]):
    dataset=pd.read_csv(path,sep=',')
    #print(dataset)
    for col in drop_list:
        dataset=dataset.drop(col,axis=1)

    return dataset

