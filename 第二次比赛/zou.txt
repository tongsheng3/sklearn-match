from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer as si
import numpy as np
import pandas as pd

#用于加载数并进行一些简单的去问号处理
def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    x = pd.DataFrame(data)
    x = x.replace('?',np.nan)
    model = si(missing_values=np.nan ,strategy='most_frequent')
    x = model.fit_transform(x).astype(np.int)
    return x


train = read_data('train.csv')
test = read_data('test.csv')
#32时效果最好
knn = KNeighborsClassifier(k=32)
knn.fit(train[:,:-1],train[:,-1])
knn.predict(test)