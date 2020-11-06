import pandas as pd
import time
from numpy import ravel, savetxt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def image_show(img):
    plt.imshow(img)
    plt.show()


def opencsv():  # 使用pandas打开
    data = pd.read_csv('data/train.csv')
    data1 = pd.read_csv('data/test.csv')
    train_x = data.values[0:, 1:]  # 读入全部训练数据
    train_y = data.values[0:, 0]    # values:查看数据表的值
    result_x = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_x, train_y, result_x


def data_pro(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=33)
    return x_train, x_test, y_train, y_test


def knnClassify(x_train, x_test, y_train, y_test):
    id = range(1, x_test.shape[0] + 1)
    print("start run knn.")
    # 训练
    knnClf = KNeighborsClassifier()  # k=5   KNN中邻值为5，
    knnClf.fit(x_train, ravel(y_train))

    # 预测
    y_predict = knnClf.predict(x_test)
    print("score on the testdata:", knnClf.score(x_test, y_test))
    # print("score on the traindata:",knnClf.score(x_train,y_train))
    print(classification_report(y_test, y_predict))
    # 可能性
    probablity = knnClf.predict_proba(x_test)
    list_pro = []
    for i in range(probablity.shape[0]):
        pro = max(list(probablity[i]))
        list_pro.append(pro)
    # 输出
    index = np.array(id).reshape((-1, 1))[:, 0:1]
    result = pd.DataFrame(np.column_stack((index.reshape(-1, 1), np.array(y_test).reshape(-1, 1),
                                           np.array(y_predict).reshape(-1, 1), np.array(list_pro).reshape(-1, 1))),
                          columns=['ImageId', 'test_label', 'predict_lable', 'probablity'])
    result.to_csv('result/knn_result.csv', index=False, header=True, encoding='gbk')

    # 错误分析
    diff_index = []
    for i in range(result.shape[0]):
        diff_index.append(result['test_label'][i] != result['predict_lable'][i])
    print(diff_index)
    diff = result[diff_index]
    diff_x = x_test_original[diff_index]

    # 查看每个错误
    for i in range(len(diff_index)):
        # print("label is:",diff['test_label'][i],"predict is:",diff['predict_lable'][i])
        print("test label is :", diff.iloc[i]['test_label'], 'predict label is :', diff.iloc[i]['predict_lable'])
        x = diff_x[i]
        img = x.reshape(28, 28)
        image_show(img)

    diff.to_csv('result/knn_result_diff.csv', index=False, header=True, encoding='gbk')


def svmClassify(train_x, train_y, test_x):
    id = range(1, 28001)
    t = time.time()
    svc = svm.SVC(kernel='rbf', C=10)
    svc.fit(train_x, train_y)
    h = time.time()
    print('time used:%f' % (h - t))
    test_y = svc.predict(test_x)
    k = time.time()

    print('time used:%f' % (k - h))
    savetxt('sklearn_svm_Result.csv', test_y, delimiter=',')
    result = pd.DataFrame(
        np.column_stack((np.array(id).reshape((-1, 1))[:, 0:1], np.array(test_y).reshape((-1, 1))[:, 0:1])),
        columns=['ImageId', 'Label'])

    result.to_csv("sklearn_knn_Result2.csv", index=False, header=True, encoding='gbk')


if __name__ == "__main__":
    print("start.")
    # 原数据
    train_x_original, train_y_original, result_x_original = opencsv()
    # 交叉验证
    x_train_original, x_test_original, y_train, y_test = data_pro(train_x_original, train_y_original)
    # 降维
    pca = PCA(n_components=0.8, whiten=True)
    train_x_pca = pca.fit_transform(x_train_original)
    x_test_pca = pca.transform(x_test_original)
    result_x_pca = pca.transform(result_x_original)
    # knn
    knnClassify(train_x_pca, x_test_pca, y_train, y_test)
    # SVM
    # svmClassify(train_x,train_y,test_x)
    print("end.")
