# 这是什么？

这是课程《大数据分析》的结课作业，利用C4.5决策树算法完成对某一个数据集的分类。
我选用的数据集是：(记录链接模式匹配数据集)[http://archive.ics.uci.edu/ml/machine-learning-databases/00210/]，来源于UCI。有关于该数据集的详细信息可以访问：(这里)[http://archive.ics.uci.edu/ml/machine-learning-databases/00210/documentation]
我使用的算法改进之前是C4.5，详细信息请WIKI以下。

# 安装
若像运行上述代码，需要先进行安装。
## 安装pyhon
推荐使用anaconda环境.
 ```
python version >=3.5
```
## 安装数据科学库
包括：
1. numpy
2. scipy
3. pandas
4. scikit-learn
5. graphviz
6. matplotlib

## 导入本项目的代码
注意，必须所有代码都被留存。

# 运行
使用如下命令：
```shell
python3 main.py
```
注意：当你使用时，必须main.py中的文件路径，你需要将文件路径修改为你准备去读取的csv数据文件 的路径
# 可视化
当运行完成时，会在main.py的相同文件夹下生成一个pdf文件，那里面有生成的决策树的所有信息。
# 评估
评估时自动进行的，结果将会输出在控制台上。
# 报错
程序没有错误。如果出现了警告不用担心，那源于对pandas的dataframe格式的小小修改，不用担心，一切正常。
# 其他
其他事请发issue，也可以联系2273067585@qq.com



