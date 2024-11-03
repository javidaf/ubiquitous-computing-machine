from ubiquitous_computing_machine.utils import load_data, dummy_encode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ubiquitous_computing_machine.plot import *
import numpy as np
import matplotlib.pyplot as plt

from ubiquitous_computing_machine.Problem1.part_1b import repeat_experiment
from ubiquitous_computing_machine.Problem1.part_1c import *
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.genmod.families import Gaussian
from sklearn.preprocessing import StandardScaler
from pygam import GAM, s
from ubiquitous_computing_machine.utils import feature_names


file = r"ubiquitous_computing_machine\data\qsar_aquatic_toxicity.csv"
X, y = load_data(file)  # (546, 8) (546,) panda data frame


def fit_gam_revised(X, y, df_spline=[4, 8], random_state=42):
    """
    Fit GAM using pygam library
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state
    )

    results = {}

    for df in df_spline:
        gam = GAM(
            s(0, n_splines=df)
            + s(1, n_splines=df)
            + s(2, n_splines=df)
            + s(3, n_splines=df)
            + s(4, n_splines=df)
            + s(5, n_splines=df)
            + s(6, n_splines=df)
            + s(7, n_splines=df)
        )

        gam.fit(X_train, y_train)

        y_train_pred = gam.predict(X_train)
        y_test_pred = gam.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        results[df] = {"model": gam, "train_mse": train_mse, "test_mse": test_mse}

    return results


def plot_gam_effects_revised(gam_model, feature_names):
    """
    Plot partial dependence plots for GAM
    """
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        XX = gam_model.generate_X_grid(term=i)
        pdep = gam_model.partial_dependence(term=i, X=XX)
        ax.plot(XX[:, i], pdep)
        ax.set_title(feature_names[i])

    plt.tight_layout()
    return fig


# Fit GAM with different complexities
gam_results = fit_gam_revised(X, y, df_spline=[4, 8])

# Print results
for df, result in gam_results.items():
    print(f"\nResults for df={df}:")
    print(f"Training MSE: {result['train_mse']:.3f}")
    print(f"Test MSE: {result['test_mse']:.3f}")

# Plot effects for one of the models (e.g., df=4)
plot_gam_effects_revised(gam_results[4]["model"], feature_names)
plt.show()
