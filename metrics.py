import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import average_precision_score as ap_score

class Metrics():
    def get_r2(self, output, label, classes):
        y_true = np.zeros( len(classes) )
        y_true[label] = label
        y_pred = output
        r2 = r2_score(y_true, y_pred)
        return r2

    def get_mse(self, output, label, classes):
        y_true = np.zeros( len(classes) )
        y_true[label] = label
        y_pred = output
        mse = mse_score(y_true, y_pred)
        return mse

    def get_ap(self, output, label, classes):
        y_true = np.zeros( len(classes) )
        y_true[label] = label
        y_pred = output
        ap = ap_score(y_true, y_pred, pos_label=label)
        return ap

#metrics = Metrics(); print(metrics.get_r2([0, 0, 2, 0, 0], [2], ['a', 'b', 'c', 'd', 'e']))       