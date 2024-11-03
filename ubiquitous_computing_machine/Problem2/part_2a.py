import time
from ubiquitous_computing_machine.utils import load_data2
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    LeaveOneOut,
)
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np


# file = r"ubiquitous_computing_machine\data\pimaindiansdiabetes2_imputed.csv"

# X, y = load_data2(file, header=1)  # (768, 8) (768,) panda data frame
# print(X.shape, y.shape)
# print(X.head())
# print(y.head()) # pos or neg


def evaluate_knn(X, y, k_values):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=42, stratify=y
    )

    cv5_errors = []
    loocv_errors = []
    test_errors = []

    time_usage_cv5 = []
    time_usage_loocv = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        start_time = time.time()
        cv5 = cross_val_score(
            knn,
            X_train,
            y_train,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="accuracy",
        )
        end_time = time.time()
        cv5_errors.append(1 - cv5.mean())
        time_usage_cv5.append(end_time - start_time)

        start_time = time.time()
        loocv = cross_val_score(
            knn,
            X_train,
            y_train,
            cv=LeaveOneOut(),
            scoring="accuracy",
        )
        end_time = time.time()
        loocv_errors.append(1 - loocv.mean())
        time_usage_loocv.append(end_time - start_time)

        knn.fit(X_train, y_train)
        test_errors.append(1 - knn.score(X_test, y_test))

    return cv5_errors, loocv_errors, test_errors, time_usage_cv5, time_usage_loocv


def plot_errors(k_values, cv5_errors, loocv_errors, test_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv5_errors, label="5-Fold CV Error", marker="o")
    plt.plot(k_values, loocv_errors, label="Leave-One-Out CV Error", marker="s")
    plt.plot(k_values, test_errors, label="Test Error", marker="^")
    plt.xlabel("Number of Neighbors k")
    plt.ylabel("Error Rate")
    plt.title("k-NN Classifier Error Rates")
    plt.legend()
    plt.grid(True)
    plt.show()
