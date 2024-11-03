import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ubiquitous_computing_machine.utils import dummy_encode


def linear_effect(X, y, r_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=r_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    coef = model.coef_
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    r2_test = model.score(X_test, y_test)
    r2_train = model.score(X_train, y_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)

    eval_metrics = {
        "train_error": mse_train,
        "test_error": mse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }
    return eval_metrics, coef


def dummy_encoding_effect(X, y, r_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=r_state
    )
    X_train_bin = dummy_encode(X_train)
    X_test_bin = dummy_encode(X_test)
    model = LinearRegression()
    model.fit(X_train_bin, y_train)
    coef = model.coef_
    y_train_pred = model.predict(X_train_bin)
    y_test_pred = model.predict(X_test_bin)
    r2_train = model.score(X_train_bin, y_train)
    r2_test = model.score(X_test_bin, y_test)
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    eval_metrics = {
        "train_error": train_error,
        "test_error": test_error,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }
    return eval_metrics, coef


# train_error, test_error, r2_train, r2_test = linear_effect()
# train_error_ii, test_error_ii, r2_train_ii, r2_test_ii = dummy_encoding_effect()

# print(train_error, test_error, r2_train, r2_test)


# plot_model_comparison(
#     train_error,
#     test_error,
#     r2_train,
#     r2_test,
#     train_error_ii,
#     test_error_ii,
#     r2_train_ii,
#     r2_test_ii,
# )


def plot_coefficients(coef, feature_names, ax, title):
    coef = np.array(coef)
    features = np.array(feature_names)

    ax.barh(features, coef)
    ax.set_xlabel("Coefficient Value")
    ax.set_title(title)
    ax.grid(True)


def plot_both_coefficients(coef1, coef2):
    feature_names = [
        "TPSA(Tot)",
        "SAacc",
        "H-050",  # discrete values
        "MLOGP",
        "RDCHI",
        "GATS1p",
        "nN",  # discrete values
        "C-040",  # discrete values
    ]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    plot_coefficients(coef1, feature_names, axes[0], "Linear Coefficients")
    plot_coefficients(coef2, feature_names, axes[1], "Linear dummy encode Coefficients")

    plt.tight_layout()
    plt.show()
