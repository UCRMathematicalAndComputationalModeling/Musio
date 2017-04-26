from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class MultiClassLogisticRegressorPurePython(object):
    def __init__(self, lr, n_iter):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.m_ = X.shape[0]
        # self.w_ = np.zeros((X.shape[1], self.n_classes_))
        self.w_ = np.random.normal(1, 0.001, ((X.shape[1], self.n_classes_)))
        self.cost_ = []

        for i in range(self.n_iter):
            z = self.net_input(X)
            assert not np.isnan(np.sum(z))
            p_y = self.softmax_fn(z)
            y_onehot = self.onehot_fn(y)
            error = (y_onehot - p_y)
            grad = (-1 / self.m_) * X.T.dot(error)
            self.w_ = self.w_ - (self.lr * grad)

            cost = (-1 / self.m_) * np.sum(y_onehot * np.log(p_y))
            self.cost_.append(cost)

        return self

    def onehot_fn(self, y):
        onehot = np.eye(self.n_classes_)[y]
        return onehot

    def net_input(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        z = self.net_input(X)
        return np.argmax(self.softmax_fn(z), axis=1)

    def softmax_fn(self, z):
        z -= np.max(z)
        softmax = (np.exp(z).T / np.sum(np.exp(z), axis=1)).T
        return softmax


# get data
all_charts = pd.read_csv('BillboardLyricData.txt', sep='\t', encoding='utf-8')
all_charts = all_charts.dropna()
print(all_charts.shape)

# countvecotrize data
n_features_to_extract_from_text = 500
vectorizer = CountVectorizer(max_df=0.5, min_df=0.0,max_features=n_features_to_extract_from_text,stop_words='english')
X = np.asarray(vectorizer.fit_transform(all_charts.lyrics).todense())
print(X.shape)

# y to ints
class_mapping = {label:idx for idx,label in enumerate(np.unique(all_charts.chart))}
y = all_charts.chart.map(class_mapping)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# scale
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)
print('X_train_std mean', X_train_std.mean())
print('X_test_std mean', X_test_std.mean())


# instantiate and fit model1
model1 = LogisticRegression(penalty='l2', C=0.3)
model1 = model1.fit(X_train_std, y_train)
# evaluate model1
train_acc = model1.score(X_train_std, y_train)
test_acc = model1.score(X_test_std, y_test)
print('Model 1')
print('Train accuracy: {}'.format(train_acc))
print('Test accuracy: {}'.format(test_acc))


# instantiate and fit model2
model2 = MultiClassLogisticRegressorPurePython(n_iter=50, lr=0.001)
model2.fit(X_train_std, y_train)
plt.plot(model2.cost_)
# plt.show()
# evaluate model2
train_ps = model2.predict(X_train_std)
train_ps = model2.predict(X_train_std)
train_acc = np.sum(train_ps == y_train) / float(len(X_train))
test_acc = np.sum(model2.predict(X_test_std) == y_test) / float(len(X_test))
print('Model 2')
print('Train accuracy: {}'.format(train_acc))
print('Test accuracy: {}'.format(test_acc))
