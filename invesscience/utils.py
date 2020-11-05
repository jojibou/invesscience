
import numpy as numpy
import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def compute_precision(y_pred, y_true):


    return precision_score(y_true, y_pred)

def compute_recall(y_pred, y_true):

     return recall_score(y_true, y_pred)

def compute_f1(y_pred, y_true):

     return f1_score(y_true, y_pred)

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
