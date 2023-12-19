import numpy as np
from statistics import mode


class KNN:
    def __init__(self, metric='minkowski', p=2, n_neighbors=5):
        metrics = {
            'minkowski': lambda X, Y: np.sqrt(np.sum(list(map(lambda x, y: abs(x - y) ** p, X, Y)))),
            'euclidean': lambda X, Y: np.sqrt(np.sum(list(map(lambda x, y: (x - y) ** 2, X, Y)))),
            'manhattan': lambda X, Y: np.sqrt(np.sum(list(map(lambda x, y: abs(x - y), X, Y))))
        }
        self.distance_func = metrics[metric]
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.index = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.indexes = np.arange(0, len(self.X_train))
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        result = list()
        for x in X_test:
            result.append(mode([self.y_train[i] for i in self.get_neighbors(x)]))
        return np.array(result)

    def get_neighbors(self, x):
        distances = dict(zip(self.indexes, [self.distance_func(x, x_train) for x_train in self.X_train]))
        k_neighbors = sorted(distances.items(), key=lambda item: item[1])[:self.n_neighbors]
        return [neighbors[0] for neighbors in k_neighbors]
