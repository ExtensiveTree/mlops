import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os


def createDirs(dirNames):
    for dirName in dirNames:
        if not os.path.exists(dirName):
            os.makedirs(dirName)


df = pd.read_csv('./train/train_normal.csv', delimiter=',')
train, quality = df.drop(columns='quality').values, df['quality'].values

x_train, x_val, y_train, y_val = train_test_split(train, quality, test_size=0.2, random_state=42)


def training(x_t, y_t, x_v, y_v, model_dir = 'model', pkl_filename = "./model/pickle_model.pkl"):
    model = LinearRegression(fit_intercept=True)
    model.fit(x_t, y_t)

    createDirs([model_dir])

    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    y_p = model.predict(x_v)

    print("Validation score: {0:.2f} %".format(100 * r2_score(y_v, y_p)))


training(x_train, y_train, x_val, y_val)




