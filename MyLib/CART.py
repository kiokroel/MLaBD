import numpy as np
from statistics import mode


class Node:
    def __init__(self, depth):
        self.value = None
        self.left = None
        self.right = None
        self.depth = depth

    '''def get_left(self):
        return self.left

    def get_right(self):
        return self.right'''


class Split:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold


class CART:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        criterions = {
            'gini': self.__gini,
            'mse': self.__mse
        }
        self.criterion = criterions[criterion]
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree_root = self.__build_children(Node(depth=0), X, y)

    def predict(self, X):
        return np.array([self.__pred(x, self.tree_root) for ind, x in X.iterrows()])

    def __build_children(self, node, X, y):
        if (self.max_depth and node.depth >= self.max_depth) or X.shape[0] <= self.min_samples_split:
            node.value = self.__find_value(y)
            return node
        split, data_split = self.__best_split(X, y)
        if y[data_split[0]].empty or y[data_split[1]].empty:
            if y[data_split[0]].empty:
                node.value = self.__find_value(y[data_split[1]])
            else:
                node.value = self.__find_value(y[data_split[0]])
            return node
        node.value = split
        node.left = self.__build_children(Node(node.depth+1), X[data_split[0]], y[data_split[0]])
        node.right = self.__build_children(Node(node.depth+1), X[data_split[1]], y[data_split[1]])
        return node

    def __best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_split = None
        best_param = float('inf')
        for feature in X.keys():
            thresholds = X[feature].unique()
            for threshold in thresholds:
                split_left, split_right = self.__split(X, feature, threshold)
                param = self.criterion(y[split_left], y[split_right])
                if param < best_param:
                    best_param = param
                    best_feature = feature
                    best_threshold = threshold
                    best_split = [split_left, split_right]
        return Split(feature=best_feature, threshold=best_threshold), best_split

    def __split(self, X, feature, threshold):
        return X.loc[:, feature] <= threshold, X.loc[:, feature] > threshold

    def __pred(self, x, node):
        if type(node.value) is not Split:
            return node.value
        feature, threshold = node.value.feature, node.value.threshold
        if x[feature] <= threshold:
            return self.__pred(x, node.left)
        else:
            return self.__pred(x, node.right)

    def __gini(self, left, right):
        left_gini = 1 - np.sum((np.bincount(left) / len(left)) ** 2)
        right_gini = 1 - np.sum((np.bincount(right) / len(right)) ** 2)
        gini = (len(left) * left_gini + len(right) * right_gini) / (len(left) + len(right))
        return gini

    def __mse(self, left, right):
        left_mse = np.mean((left - np.mean(left)) ** 2)
        right_mse = np.mean((right - np.mean(right)) ** 2)
        mse = (len(left) * left_mse + len(right) * right_mse) / (len(left) + len(right))
        return mse

    def __find_value(self, y):
        if self.criterion == self.__mse:
            return np.mean(y)
        if self.criterion == self.__gini:
            return mode(y.loc[:y.keys()[0]])
