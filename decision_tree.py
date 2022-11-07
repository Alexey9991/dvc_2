from sklearn.tree import DecisionTreeClassifier

from utils import get_data, save_data

if __name__ == "__main__":
    x_train, y_train = get_data("data/dataset/iris_train.data", "target_class")
    x_test, y_test = get_data("data/dataset/iris_test.data", "target_class")

    tree = DecisionTreeClassifier().fit(x_train, y_train)
    tree = tree.fit(x_train, y_train)

    y_train_pred = tree.predict(x_train)
    y_test_pred = tree.predict(x_test)

    save_data("data/predictions/decision_tree_train.txt", y_train_pred)
    save_data("data/predictions/decision_tree_test.txt", y_test_pred)
