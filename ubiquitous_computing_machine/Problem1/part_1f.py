import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
from ubiquitous_computing_machine.utils import feature_names


# # Load the data
# file = r"ubiquitous_computing_machine\data\qsar_aquatic_toxicity.csv"
# X, y = load_data(file)  # (546, 8) (546,) panda data frame


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42
# )


def fit_decision_tree(X_train, y_train):
    reg_tree = DecisionTreeRegressor(random_state=42)
    reg_tree.fit(X_train, y_train)
    return reg_tree


def get_pruning_path(reg_tree, X_train, y_train):
    path = reg_tree.cost_complexity_pruning_path(X_train, y_train)
    return path.ccp_alphas, path.impurities


def plot_pruning_path(ccp_alphas, impurities):
    plt.figure(figsize=(8, 4))
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    plt.xlabel("Effective Alpha")
    plt.ylabel("Total Impurity of Leaves")
    plt.title("Cost-Complexity Pruning Path")
    plt.show()


def train_trees(X_train, y_train, ccp_alphas):
    trees = []
    for alpha in ccp_alphas:
        tree = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees


def evaluate_trees(trees, X_train, y_train, X_test, y_test, ccp_alphas):
    train_scores = [tree.score(X_train, y_train) for tree in trees]
    test_scores = [tree.score(X_test, y_test) for tree in trees]
    mse_train = [np.mean((tree.predict(X_train) - y_train) ** 2) for tree in trees]
    mse_test = [np.mean((tree.predict(X_test) - y_test) ** 2) for tree in trees]

    plt.figure(figsize=(8, 4))
    plt.plot(
        ccp_alphas, train_scores, marker="o", label="Train", drawstyle="steps-post"
    )
    plt.plot(ccp_alphas, mse_test, marker="o", label="Test", drawstyle="steps-post")
    plt.xlabel("Effective Alpha")
    plt.ylabel("MSE")
    plt.title("Performance vs Effective Alpha")
    plt.legend()
    plt.show()

    eval_metrics = {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "mse_train": mse_train,
        "mse_test": mse_test,
    }

    return eval_metrics


def select_optimal_tree(ccp_alphas, test_scores, trees):
    optimal_index = np.argmax(test_scores)
    optimal_alpha = ccp_alphas[optimal_index]
    optimal_tree = trees[optimal_index]

    print(f"Optimal alpha: {optimal_alpha}")
    print(f"Optimal tree depth: {optimal_tree.get_depth()}")
    print(f"Optimal number of leaves: {optimal_tree.get_n_leaves()}")

    plt.figure(figsize=(20, 10))
    plot_tree(
        optimal_tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=6,
        max_depth=2,
    )
    plt.show()

    return optimal_tree
