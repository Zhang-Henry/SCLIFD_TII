# -*- coding: utf-8 -*-
"""
Created on July 30 2022

@author: zhr
"""
from sklearn import preprocessing
import numpy as np

def get_data():
    train_dataset = ['../origin_data/d01.txt','../origin_data/d02.txt','../origin_data/d04.txt','../origin_data/d06.txt','../origin_data/d07.txt','../origin_data/d08.txt','../origin_data/d12.txt','../origin_data/d14.txt','../origin_data/d18.txt']
    test_dataset = ['../origin_data/d01_te.txt','../origin_data/d02_te.txt','../origin_data/d04_te.txt','../origin_data/d06_te.txt','../origin_data/d07_te.txt','../origin_data/d08_te.txt','../origin_data/d12_te.txt','../origin_data/d14_te.txt','../origin_data/d18_te.txt']

    positive_data = np.loadtxt('../origin_data/d00.txt')
    positive_data = positive_data.T

    # positive_data = positive_data[range(0,500,10),:] #
    negative_data = np.empty(shape=[0, 52])
    for i in range(9):
        tmp = np.loadtxt(train_dataset[i])

        tmp = tmp[range(0,480,24),:]#每个类取20个故障样本
        negative_data = np.concatenate((negative_data, tmp))

    data = np.concatenate((positive_data, negative_data))
    scaler = preprocessing.MinMaxScaler().fit(data)
    data = scaler.transform(data)

    X_train = []
    X_train.append(data)
    y_train = np.zeros(500)
    for i in range(1,10):
        y_train = np.concatenate((y_train,[i]*20)) #训练集：1-9类为非正常类，每一类分别有20个样本

    X_train = np.array(X_train)

    positive_data = np.loadtxt('../origin_data/d00_te.txt')

    negative_data = np.empty(shape=[0, 52])
    for i in range(9):
        tmp = np.loadtxt(test_dataset[i])
        negative_data = np.concatenate((negative_data, tmp[160:,:]))

    data = np.concatenate((positive_data, negative_data))
    data = scaler.transform(data)
    X_test = []

    X_test.append(data)
    y_test = []

    y_test = np.zeros(960) #测试集：0类-正常类有960个样本
    for i in range(1,10):
        y_test=np.concatenate((y_test,[i]*800)) #测试集：1-9类为非正常类，每一类分别有800个样本

    X_test = np.array(X_test)
    X_train = np.reshape(X_train,(-1,52))
    X_test = np.reshape(X_test,(-1,52))


    np.save('long_tail/X_train.npy', X_train)
    np.save('long_tail/y_train.npy', y_train)
    np.save('long_tail/X_test.npy', X_test)
    np.save('long_tail/y_test.npy', y_test)
    return [X_train, y_train, X_test, y_test]



if __name__ == '__main__':
    get_data()