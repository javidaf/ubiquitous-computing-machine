from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pygam import GAM, s


def fit_gam(X, y, df_spline=[4, 8], random_state=42, spline_type=3):
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
            s(0, n_splines=df, spline_type=spline_type)
            + s(1, n_splines=df, spline_type=spline_type)
            + s(2, n_splines=df, spline_type=spline_type)
            + s(3, n_splines=df, spline_type=spline_type)
            + s(4, n_splines=df, spline_type=spline_type)
            + s(5, n_splines=df, spline_type=spline_type)
            + s(6, n_splines=df, spline_type=spline_type)
            + s(7, n_splines=df, spline_type=spline_type)
        )

        gam.fit(X_train, y_train)

        y_train_pred = gam.predict(X_train)
        y_test_pred = gam.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        results[df] = {"model": gam, "train_mse": train_mse, "test_mse": test_mse}

    return results


def plot_gam_effects(gam_model, feature_names):
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
