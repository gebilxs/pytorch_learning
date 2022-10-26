import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
x = np.linspace(data.Population.min(), data.Population.max(), 100)# 画线用的 这里是一维数组
f = g[0, 0] + (g[0, 1] * x) #一维矩阵的每个元素乘以参数 这里也表示了函数关系
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
#散点图 scatter 零散的 plot 图
ax.scatter(data.Population, data.Profit, label='Traning Data')
#图例位置编号
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
