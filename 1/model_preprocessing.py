import pandas as pd
from sklearn import preprocessing

train = pd.read_csv('./train/train.csv', delimiter=';')
test = pd.read_csv('./test/test.csv', delimiter=';')


def data_preprocessing(data, pathname):
    scaler = preprocessing.MinMaxScaler()
    df = scaler.fit_transform(data)
    name = data.columns
    data_norm = pd.DataFrame(df, columns=name)
    data_norm.to_csv(pathname, index=False)


data_preprocessing(train, './train/train_normal.csv')

data_preprocessing(test, './test/test_norm.csv')
