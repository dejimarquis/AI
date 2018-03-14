import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import quandl
import matplotlib.pyplot as plt

class CryptoCurrencies:
    def __init__(self):
        ybcdata = np.array(quandl.get("BCHARTS/BITSTAMPUSD.7", start_date="2017-08-01").get_values())
        xtime = np.array([-ybcdata.shape[0]-1+i for i in range(ybcdata.shape[0])]).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(xtime, ybcdata, random_state=66)
        linear = LinearRegression()
        linear.fit(X_train, y_train)
        print(linear.score(X_train, y_train))
        print(linear.score(X_test, y_test))
        yresult_test=linear.predict(X_test)
        X_new = np.array([i for i in range(31)]).reshape(-1, 1)
        yresult_predict=linear.predict(X_new)
        print(yresult_predict)
        plt.plot(xtime,ybcdata, label='true')
        # plt.scatter(X_train, y_train, label='train')
        plt.plot(X_test, yresult_test, label='test')
        plt.plot(X_new, yresult_predict, label='predictions')
        plt.legend()
        plt.savefig('./Q1.png')
        plt.close()

if __name__ =='__main__':
    CryptoCurrencies()