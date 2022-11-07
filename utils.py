import numpy as np
import pandas as pd

CLUSTER_NUM = 3
IRIS_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal width", "target_class"]


def get_data(path, target_column, names=IRIS_COLUMNS):
    data = pd.read_csv(path, names=names)
    x = data.loc[:, data.columns != target_column]
    y = data[target_column]
    return x, y


def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def get_relation(y_true, y_pred, cluster_num=CLUSTER_NUM):
    relation = dict()
    for cluster in range(cluster_num):
        mode = mode1(y_true.values[np.where(y_pred == cluster)])[0]
        relation[mode] = cluster
    return relation


def save_data(path, data, prefix=""):
    with open(path, "w") as f:
        for val in data:
            f.write(prefix + str(val) + "\n")


def replace_with_dict2_generic(ar, dic, assume_all_present=True):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    idx = np.searchsorted(ks,ar)

    if not assume_all_present:
        idx[idx == len(vs)] = 0
        mask = ks[idx] == ar
        return np.where(mask, vs[idx], ar)
    else:
        return vs[idx]
