from sklearn import preprocessing
import numpy as np
def get_data():
    train_dataset = ['./d01.txt','./d02.txt','./d04.txt','./d06.txt','./d07.txt','./d08.txt','./d12.txt','./d14.txt','./d18.txt']
    test_dataset = ['./d01_te.txt','./d02_te.txt','./d04_te.txt','./d06_te.txt','./d07_te.txt','./d08_te.txt','./d12_te.txt','./d14_te.txt','./d18_te.txt']

    positive_data = np.loadtxt('./d00.txt')
    positive_data = positive_data.T
    negative_data = np.empty(shape=[0, 52])

    for i in range(9):
        tmp = np.loadtxt(train_dataset[i])
        negative_data = np.concatenate((negative_data, tmp))


    data = np.concatenate((positive_data, negative_data))
    scaler = preprocessing.MinMaxScaler().fit(data)
    data = scaler.transform(data)

    X_train = []
    X_train.append(data)
    y_train = []
    for index in range(500):
        y_train.append(0)
    for i in range(9):
        for index in range(480):
             y_train.append(i+1)


    X_train = np.array(X_train)

    y_train = np.array(y_train)


    positive_data = np.loadtxt('./d00_te.txt')

    negative_data = np.empty(shape=[0, 52])
    for i in range(9):
        tmp = np.loadtxt(test_dataset[i])
        negative_data = np.concatenate((negative_data, tmp[160:,:]))

    data = np.concatenate((positive_data, negative_data))
    data = scaler.transform(data)
    X_test = []

    X_test.append(data)
    y_test = []

    for index in range(960):
        y_test.append(0)
    for i in range(9):
        for index in range(960+i*800,960+(i+1)*800):
            y_test.append(i+1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train = np.reshape(X_train,(-1,52))

    X_test = np.reshape(X_test,(-1,52))

    np.save('X_train_multiclass_Balanced.npy', X_train)
    np.save('y_train_multiclass_Balanced.npy', y_train)
    np.save('X_test_multiclass_Balanced.npy', X_test)
    np.save('y_test_multiclass_Balanced.npy', y_test)
    return [X_train, y_train, X_test, y_test]

if __name__ == '__main__':
    get_data()
