import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

#%matplotlib inline #如果你使用的是jupyter notebook取消注释
# np.random.seed(1)
#
# y_hat = tf.constant(36,name="y_hat")            #定义y_hat为固定值36
# y = tf.constant(39,name="y")                    #定义y为固定值39
#
# loss = tf.Variable((y-y_hat)**2,name="loss" )   #为损失函数创建一个变量
#
# init = tf.compat.v1.global_variables_initializer()        #运行之后的初始化(ession.run(init))
#
#
#                                                 #损失变量将被初始化并准备计算
# with tf.compat.v1.Session() as session:                   #创建一个session并打印输出
#     session.run(init)                           #初始化变量
#     print(session.run(loss))                    #打印损失值
def linear_function():
    """
    实现一个线性功能：
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    返回：
        result - 运行了session后的结果，运行的是Y = WX + b

    """

    np.random.seed(1)  # 指定随机种子

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul是矩阵乘法
    # Y = tf.matmul(W,X) + b #也可以以写成这样子

    # 创建一个session并运行它
    sess = tf.compat.v1.Session()
    result = sess.run(Y)

    # session使用完毕，关闭它
    sess.close()

    return result
print("result = " +  str(linear_function()))

## didn't finish