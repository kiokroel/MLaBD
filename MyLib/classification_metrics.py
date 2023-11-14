import numpy as np


def accuracy(y_test, y_pred):
    right = 0
    y_test = np.array(y_test)
    for i in range(len(y_test)):
        if(y_test[i] == y_pred[i]):
            right += 1
    return right/len(y_test)


def precision(y_test, y_pred):
    right_pred = 0
    y_test = np.array(y_test)
    for i in range(len(y_pred)):
        if (y_test[i] == y_pred[i] == 1):
            right_pred += 1
    positive = list(filter(lambda x: x == 1, y_pred))
    return right_pred / len(positive)


def recall(y_test, y_pred):
    right_pred = 0
    y_test = np.array(y_test)
    tp_fn = 0
    for i in range(len(y_pred)):
        if (y_test[i] == y_pred[i] == y_pred[i] == 1):
            right_pred += 1
        if (y_test[i] == 1):
            tp_fn += 1
    return right_pred / tp_fn


def f1(y_test, y_pred):
    pr = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    return 2 * pr * rec / (pr + rec)
