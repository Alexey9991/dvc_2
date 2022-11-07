import json
from sklearn.cluster import KMeans

from utils import CLUSTER_NUM, get_relation, get_data, save_data, replace_with_dict2_generic

if __name__ == "__main__":
    x_train, y_train = get_data("data/dataset/iris_train.data", "target_class")
    x_test, y_test = get_data("data/dataset/iris_test.data", "target_class")

    k_means = KMeans(n_clusters=CLUSTER_NUM)
    k_means = k_means.fit(x_train, y_train)  # second argument ignored

    relation = get_relation(y_train, k_means.labels_)

    y_train_pred = k_means.predict(x_train)
    y_test_pred = k_means.predict(x_test)

    reverse_relation = {y: x for x, y in relation.items()}
    y_train_pred = replace_with_dict2_generic(y_train_pred, reverse_relation)
    y_test_pred = replace_with_dict2_generic(y_test_pred, reverse_relation)

    save_data("data/predictions/k_means_train.txt", y_train_pred)
    save_data("data/predictions/k_means_test.txt", y_test_pred)

    with open("data/relations/k_means_relation.json", "w") as f:
        json.dump(relation, f, indent=4)
