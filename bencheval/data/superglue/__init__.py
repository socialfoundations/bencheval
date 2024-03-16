import os
import pandas as pd
import numpy as np


def load_superglue():
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaderboard.tsv"), sep="\t")
    ori_cols = data.columns[5:-2]
    cols = []
    for c in ori_cols:
        if type(data[c].values[0]) is str and "/" in data[c].values[0]:
            c1 = c + "-a"
            c2 = c + "-b"
            res1, res2 = [], []
            for line in data[c].values:
                s = line.strip().split("/")
                res1.append(float(s[0]))
                res2.append(float(s[1]))
            res1 = np.array(res1)
            res2 = np.array(res2)
            data[c1] = res1
            data[c2] = res2
            data[c] = (res1 + res2) / 2
            cols.append(c)
        else:
            cols.append(c)

    return data, cols


def test():
    data, cols = load_superglue()
    print(data.head())
    print(cols)


if __name__ == '__main__':
    test()
