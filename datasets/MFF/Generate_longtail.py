# -*- coding: utf-8 -*-
"""
Created on July 30 2022

@author: zhr
"""


from sklearn import preprocessing
import numpy as np
def get_data():
    train_dataset = ['origin/d01.txt','origin/d02.txt','origin/d03.txt','origin/d05.txt','origin/d04.txt']
    test_dataset = ['origin/d01_te.txt','origin/d02_te.txt','origin/d03_te.txt','origin/d05_te.txt','origin/d04_te.txt']

############## 生成训练数据X_train和标签y_train ##############
    positive_data = np.loadtxt('origin/d00.txt')
    # positive_data = positive_data.T
    negative_data = np.empty(shape=[0, 23])

    for i in range(5):
        tmp = np.loadtxt(train_dataset[i])
        tmp = tmp[0:500:100] #每个故障类取5个
        negative_data = np.concatenate((negative_data, tmp))

    data = np.concatenate((positive_data[:200], negative_data))
    scaler = preprocessing.MinMaxScaler().fit(data)
    data = scaler.transform(data)

    X_train = np.array(data)
    y_train = np.zeros(200)
    for i in range(1,6):
        y_train = np.concatenate((y_train,[i]*5))


############## 生成测试数据X_test和标签y_test ##############
    positive_data = np.loadtxt('origin/d00_te.txt')
    negative_data = np.empty(shape=[0, 23])
    for i in range(5):
        tmp = np.loadtxt(test_dataset[i])
        negative_data = np.concatenate((negative_data, tmp[:800])) #测试数据的故障类取800个

    data = np.concatenate((positive_data[:800], negative_data)) #测试数据的正常类取800个
    data = scaler.transform(data)
    X_test = []

    X_test.append(data)
    y_test = np.zeros(800) #测试集：0类-正常类有800个样本
    for i in range(1,6):
        y_test=np.concatenate((y_test,[i]*800)) #测试集：1-9类为非正常类，每一类分别有800个样本

    X_test = np.array(X_test)
    X_train = np.reshape(X_train,(-1,23))
    X_test = np.reshape(X_test,(-1,23))

    np.save('long_tail/X_train.npy', X_train)
    np.save('long_tail/y_train.npy', y_train)
    np.save('long_tail/X_test.npy', X_test)
    np.save('long_tail/y_test.npy', y_test)


    return [X_train, y_train, X_test, y_test]

if __name__ == '__main__':
    get_data()
