
import numpy as numpy
import time



def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def compute_precision(y_pred, y_true):

    predicted = 0
    actually = 0

    for(idx, y_p) in enumerate(y_pred):
        if y_p ==1:
            predicted +=1
            if y_true[idx]==1:
                actually +=1
    if predicted ==0:
        return 0
    else:
        return actually/predicted

def compute_recall(y_pred, y_true):

    predicted = 0
    actually = 0

    for(idx, y_p) in enumerate(y_pred):
        if y_p ==1:
            predicted +=1
            if y_true[idx]==1:
                actually +=1
    if (2*actually - predicted) ==0:
        return 0
    else:
        return actually/(2*actually - predicted)


def compute_f1(y_pred, y_true):

    recall = compute_recall(y_pred, y_true)
    precision = compute_precision(y_pred, y_true)

    return (2*precision*recall)/(precision+recall)


_true = [0,0,1,2,2,1]
_pred = [0,1,1,2,1,0]

my_custom_metric(_true, _pred)
