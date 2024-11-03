import itertools
import numpy as np
from sklearn.calibration import LabelEncoder
from ubiquitous_computing_machine.utils import load_data2
from pygam import LogisticGAM, s
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from ubiquitous_computing_machine.utils import feature_names_p2
from pygam import LogisticGAM
import pandas as pd


def fit_gam_model(X, y, gam_splines):
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = np.asarray(X, dtype=np.float16)
    y = np.asarray(y, dtype=np.float16)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=42, stratify=y
    )

    gamp_p = LogisticGAM(gam_splines).fit(X_train, y_train)
    error_rate = 1 - gamp_p.accuracy(X_test, y_test)
    return gamp_p, error_rate


def plot_p_dep_LGAM(gamp_p):
    pdep = []

    for i, term in enumerate(gamp_p.terms):
        if term.isintercept:
            continue

        XX = gamp_p.generate_X_grid(term=i)
        pdep.append(gamp_p.partial_dependence(term=i, X=XX))

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs = axs.ravel()

    for i, ax in enumerate(axs):
        ax.plot(gamp_p.generate_X_grid(term=i), pdep[i])
        ax.set_title(feature_names_p2[i])

    plt.tight_layout()
    plt.show()


def create_gam_splines(n_features, spline_order=2, n_splines=5):
    terms = [
        s(i, n_splines=n_splines, spline_order=spline_order) for i in range(n_features)
    ]
    return terms[0] + sum(terms[1:], start=terms[0])


def perform_variable_selection_logistic(X, y, spline_order=3, n_splines=5):

    max_features = X.shape[1]
    best_aic = np.inf
    best_model = None
    best_features = []
    error_rate_for_best_feature = None

    for k in range(1, max_features + 1):
        for subset in itertools.combinations(X.columns, k):
            gam_splines = create_gam_splines(
                k, spline_order=spline_order, n_splines=n_splines
            )
            gam, e_rate = fit_gam_model(X[list(subset)], y, gam_splines)
            aic = gam.statistics_["AIC"]
            if aic < best_aic:
                best_aic = aic
                best_model = gam
                best_features = subset
                error_rate_for_best_feature = e_rate
    return best_model, best_features, best_aic, error_rate_for_best_feature
