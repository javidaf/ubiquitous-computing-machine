import numpy as np
from ubiquitous_computing_machine.utils import load_data2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

# # Load data
# file = r"ubiquitous_computing_machine\data\pimaindiansdiabetes2_imputed.csv"
# X, y = load_data2(file, header=1)

# # Split data into training and test sets with stratification
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=1 / 3, random_state=42, stratify=y
# )


def fit_classification_tree(X_train, y_train, X_test, y_test, max_depth=7):
    """Fit a classification tree and compute training and test errors."""
    clf_tree = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    clf_tree.fit(X_train, y_train)

    train_error = 1 - clf_tree.score(X_train, y_train)
    test_error = 1 - clf_tree.score(X_test, y_test)

    return clf_tree, train_error, test_error


def fit_bagging_classifier(X_train, y_train, X_test, y_test, max_depth=7):
    """Fit a bagging classifier and compute training and test errors."""
    bagging = BaggingClassifier(
        DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=100,
        random_state=42,
    )
    bagging.fit(X_train, y_train)

    train_error = 1 - bagging.score(X_train, y_train)
    test_error = 1 - bagging.score(X_test, y_test)

    return bagging, train_error, test_error


def fit_random_forest(X_train, y_train, X_test, y_test, max_depth=7):
    """Fit a random forest classifier and compute training and test errors."""
    random_forest = RandomForestClassifier(
        n_estimators=100, random_state=42, max_depth=max_depth
    )
    random_forest.fit(X_train, y_train)

    train_error = 1 - random_forest.score(X_train, y_train)
    test_error = 1 - random_forest.score(X_test, y_test)

    return random_forest, train_error, test_error



def find_best_depth(X_train, y_train, X_test, y_test, max_depths):

    best_depth_tree = None
    best_depth_forest = None
    best_depth_boost = None

    best_error_tree = np.inf
    best_error_forest = np.inf
    best_error_boost = np.inf

    for d in max_depths:
        _, _, test_error_tr = fit_classification_tree(
            X_train, y_train, X_test, y_test, d
        )
        _, _, test_error_bg = fit_bagging_classifier(
            X_train, y_train, X_test, y_test, d
        )
        _, _, test_error_fo = fit_random_forest(X_train, y_train, X_test, y_test, d)

        if test_error_tr < best_error_tree:
            best_error_tree = test_error_tr
            best_depth_tree = d

        if test_error_bg < best_error_forest:
            best_error_forest = test_error_bg
            best_depth_forest = d

        if test_error_fo < best_error_boost:
            best_error_boost = test_error_fo
            best_depth_boost = d

    # results = [
    #     ["Model", "Best Depth", "Best Error"],
    #     ["Classification Tree", best_depth_tree, f"{best_error_tree:.4f}"],
    #     ["Random Forest", best_depth_forest, f"{best_error_forest:.4f}"],
    #     ["Bagging", best_depth_boost, f"{best_error_boost:.4f}"]
    # ]

    # print(tabulate(results, headers="firstrow", tablefmt="grid"))
    df_results = pd.DataFrame(
        {
            "Model": ["Classification Tree", "Random Forest", "Bagging"],
            "Best Depth": [best_depth_tree, best_depth_forest, best_depth_boost],
            "Best Error": [best_error_tree, best_error_forest, best_error_boost],
        }
    )

    print(df_results)
