# from sklearn.datasets import load_breast_cancer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
#
# import matplotlib.pyplot as plt
#
# cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target, random_state=66)
# # knn = KNeighborsClassifier()
# # knn.fit(X_train, y_train)
# # print(knn.score(X_train, y_train))
# # print(knn.score(X_test, y_test))
# train_accuracy = []
# test_accuracy = []
# for i in range(1,11):
#     knn = KNeighborsClassifier(i)
#     knn.fit(X_train, y_train)
#     train_accuracy.append(knn.score(X_train, y_train))
#     test_accuracy.append(knn.score(X_test, y_test))
#
# plt.plot(range(1,11),train_accuracy, label='train')
# plt.plot(range(1,11),test_accuracy, label='test')

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import quandl

import matplotlib.pyplot as plt

cancer = load_breast_cancer()
bcdata = quandl.get("BCHARTS/BITSTAMPUSD.7")

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target, random_state= 42)
log = LogisticRegression()
log.fit(X_train, y_train)
print(log.score(X_train, y_train))
print(log.score(X_test, y_test))
# train_accuracy = []
# test_accuracy = []
# for i in range(1,11):
#     knn = KNeighborsClassifier(i)
#     knn.fit(X_train, y_train)
#     train_accuracy.append(knn.score(X_train, y_train))
#     test_accuracy.append(knn.score(X_test, y_test))
#
# plt.plot(range(1,11),train_accuracy, label='train')
# plt.plot(range(1,11),test_accuracy, label='test')