from math import sqrt
import numpy as np


def mae(test, pred):
    _mae = 0
    for i in range(len(test)):
        _mae += abs(test[i] - pred[i])
    _mae = _mae/len(test)
    return _mae


def mse(test, pred):
    _mse = 0
    for i in range(len(test)):
        _mse += (test[i] - pred[i])**2
    _mse = _mse/len(test)
    return _mse


def rmse(test, pred):
    _mse = mse(test, pred)
    return sqrt(_mse)


def mape(test, pred):
    _mape = 0
    for i in range(len(test)):
        _mape += abs(test[i] - pred[i]) / abs(test[i])
    _mape = _mape/len(test)
    return _mape


def r2(test, pred):
    _mean = test.mean()
    r_ch = 0
    r_zn = 0
    for i in range(len(test)):
        r_ch += (test[i] - pred[i])**2
        r_zn += (test[i] - _mean) ** 2
    _r2 = 1 - r_ch / r_zn
    return _r2
