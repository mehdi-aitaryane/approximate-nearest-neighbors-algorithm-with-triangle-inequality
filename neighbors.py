import numpy as np
from distances import euclidean_distance
from metrics import accuracy, r2

class ANNeighborClassifier:

    def __init__(self, dist = euclidean_distance):
        self.dist = dist

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        temp = np.zeros(num_train)
        y = np.zeros(num_test)

        for i in range(num_test):
            distances = np.full(num_train, np.inf)
            for j in range(num_train):
                if i > 0 and j > 0:
                    if (temp[j] + distances[j-1]) < temp[j-1]:
                        continue
                distances[j] = self.dist(X[i], self.X[j])

            idx = np.argmin(distances)
            y[i] = self.y[idx]
            temp = distances    
        return y

    def score(self, X, y, metric = accuracy):
        y_pred = self.predict(X)
        return metric(y, y_pred)

class ANNeighborRegressor:

    def __init__(self, dist = euclidean_distance):
        self.dist = dist

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        num_test = X.shape[0]
        num_train = self.X.shape[0]
        temp = np.zeros(num_train)
        y = np.zeros(num_test)

        for i in range(num_test):
            distances = np.full(num_train, np.inf)
            for j in range(num_train):
                if i > 0 and j > 0:
                    if (temp[j] + distances[j-1]) < temp[j-1]:
                        continue
                distances[j] = self.dist(X[i], self.X[j])

            idx = np.argmin(distances)
            y[i] = self.y[idx]
            temp = distances
        return y

    def score(self, X, y, metric = r2):
        y_pred = self.predict(X)
        return metric(y, y_pred)