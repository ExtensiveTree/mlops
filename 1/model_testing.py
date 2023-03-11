import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
pickled_model = pickle.load(open("./model/pickle_model.pkl", 'rb'))
norm_test = pd.read_csv('./test/test_norm.csv', delimiter=',')
x, y = norm_test.drop(columns='quality').values, norm_test['quality'].values
y_predict = pickled_model.predict(x)
print("TestS score: {0:.2f} %".format(100 * r2_score(y,y_predict)))