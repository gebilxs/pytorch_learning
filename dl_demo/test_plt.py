import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体

plt.xlabel('x')
# plt.ylabel('')
plt.xlim(xmax=10, xmin=0)
plt.ylim(ymax=2, ymin=0)
x1 = []  # 自定义点
y1 = []  # 自定义点
x2 = []  # 自定义点
y2 = []  # 自定义点

colors1 = 'r'  # 点的颜色
colors2 = 'g'
area = np.pi * 4 ** 2  # 点面积
plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='a')
plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='b')
# plt.plot([0,9.5],[9.5,0],linewidth = '0.5',color='#000000')
plt.legend()
plt.yticks(())
plt.title('test')
plt.show()
