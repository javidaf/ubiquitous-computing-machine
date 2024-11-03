import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import matplotlib.pyplot as plt


def ridge_with_cv(X, y, alphas, cv_folds=5):
    ridge = Ridge()
    param_grid = {"alpha": alphas}
    grid_search = GridSearchCV(
        ridge, param_grid, cv=cv_folds, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)
    best_alpha = grid_search.best_params_["alpha"]
    cv_mse = -grid_search.best_score_
    cv_results = grid_search.cv_results_
    return best_alpha, cv_mse, cv_results


def ridge_with_bootstrap(X, y, alphas, n_bootstrap=100):
    bootstrap_mse = {alpha: [] for alpha in alphas}
    for _ in range(n_bootstrap):
        X_res, y_res = resample(X, y)
        for alpha in alphas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_res, y_res)
            y_pred = ridge.predict(X)
            mse = mean_squared_error(y, y_pred)
            bootstrap_mse[alpha].append(mse)
    avg_bootstrap_mse = {alpha: np.mean(mse) for alpha, mse in bootstrap_mse.items()}
    return avg_bootstrap_mse


def plot_results(alphas, cv_mse, bootstrap_mse):
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, cv_mse, label="Cross-Validation MSE", marker="o")
    plt.plot(
        alphas,
        [bootstrap_mse[alpha] for alpha in alphas],
        label="Bootstrap MSE",
        marker="s",
    )
    plt.xlabel("Alpha")
    plt.ylabel("Mean Squared Error")
    plt.title("Ridge Regression: CV vs Bootstrap MSE")
    plt.legend()
    plt.xscale("log")
    plt.grid(True)
    plt.show()
