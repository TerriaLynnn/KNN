import numpy as np
from networkx.drawing.tests.test_pylab import plt

"""
Parameters:
    无
Returns:
    group - 数据集
	labels - 分类标签
"""


# 函数说明:创建数据集
def createDataSet():
    # 六组二维特征
    group = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    # 六组特征的标签
    labels = ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片']
    return group, labels


def showDataSet(dataMat, labelMat):
    data_plus = []
    for i in range(len(group)):
        data_plus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 散点图
    plt.show()


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 打印数据集
    print(group)
    print(labels)
