import numpy as np

#knn模型
def knn_forecast(data, test, k):
    y = list()
    for i in test:
        x = (data[:,:-1] - i)**2
        x = np.sum(x, axis=1)
        x = np.array([x,data[:,-1]])
        x = x[:,x[0,:].argsort()]
        y.append(sum(x[1,:k]))
    return sum(np.mat(y)>k/2)

data = np.loadtxt('HTRU_2_train.csv', delimiter=',')
test = np.loadtxt('HTRU_2_test.csv', delimiter=',')

knn = knn_forecast(data, test, 11)
#将结果保存在CSV文件中
# knn = np.array(knn).reshape(700,1)
# np.savetxt('test.csv',knn)