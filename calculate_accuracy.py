from sklearn.metrics import accuracy_score

from utils import get_data

if __name__ == "__main__":
    _, y_train = get_data("data/dataset/iris_train.data", "target_class")
    _, y_test = get_data("data/dataset/iris_test.data", "target_class")

    for y_true, name in zip([y_train, y_test], ["train", "test"]):
        for alg in ["k_means", "decision_tree"]:
            with open(f"data/predictions/{alg}_{name}.txt", "r") as f:
                y_pred = [line.rstrip() for line in f]
            acc = accuracy_score(y_true, y_pred)
            with open(f"data/metrics/results.txt", "a+") as f:
                f.write(f"{alg} {name}: {acc}\n")
