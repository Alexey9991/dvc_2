import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv("data/dataset/iris.data", header=None)
    train, test = train_test_split(data, train_size=0.75)
    train.to_csv('data/dataset/iris_train.data', index=False, header=False)
    test.to_csv('data/dataset/iris_test.data', index=False, header=False)
