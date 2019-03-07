import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

#%%
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 文件路径
# print(os.getcwd()+'\\iris.data')
path = str(os.getcwd() + '\\iris.data')
# dtype规定类型，通过其他来分割就用delimiter，converters对数据进行预处理（这里是对第四行数据进行
# iris_type函数处理
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
# 前四列是数据，第四列才是类型即y
x, y = np.split(data, (4,), axis=1)
# 为了可视化方便，仅适用前两列特征
x = x[:, :2]
# sklearn.train_test_split把数据集分割成训练集和测试集，
# test_size是指测试集在整个数据集中所占的比例
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# ss = StandardScaler()
# x = ss.fit_transform(x)
# model = DecisionTreeClassifier(criterion='entropy', max_depth=5)
# model.fit(x_test, y_test)
# 等于下面
model = Pipeline([
    # 归一化，特征均值都是0，方程都是1
    ('ss', StandardScaler()),
    ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))
])
model = model.fit(x_train, y_train)
y_test_hat = model.predict(x_test)
#%%
# 保存 决策树可视化文件
# dot -Tpng -o 1.png 1.dot
# f = open('.\\iris_tree.dot', 'w')
# # 需要从管道中读取decisiontree数据，用get_params
# tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)
with open('.\\iris_tree.dot', 'w') as f:
    tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)

#%% 画图
N, M = 100, 100
# 横坐标和纵坐标最大最小值
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
x2_min, x2_max = x[:, 1].min(), x[:, 0].max()
# 等间隔生成数值，默认50
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, M)
# 生成网格采样点，从坐标返回坐标矩阵
x1, x2 = np.meshgrid(t1, t2)

x_show = np.stack((x1.flat, x2.flat), axis=1)