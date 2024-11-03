import statsmodels.api as sm
from ubiquitous_computing_machine.utils import load_data
from sklearn.metrics import mean_squared_error


def forward_selection(X, y, criterion="AIC"):
    initial_features = []
    best_features = initial_features.copy()
    remaining_features = X.columns.tolist()

    X_model = sm.add_constant(X[best_features])
    model = sm.OLS(y, X_model).fit()
    current_score = model.aic if criterion == "AIC" else model.bic

    while remaining_features:
        scores_with_candidates = []
        for candidate in remaining_features:
            features = best_features + [candidate]
            X_model = sm.add_constant(X[features])
            model = sm.OLS(y, X_model).fit()
            score = model.aic if criterion == "AIC" else model.bic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if best_new_score < current_score:
            remaining_features.remove(best_candidate)
            best_features.append(best_candidate)
            current_score = best_new_score
        else:
            break

    beat_featured_model = sm.OLS(y, sm.add_constant(X[best_features])).fit()
    predictions = beat_featured_model.predict(sm.add_constant(X[best_features]))
    mse = mean_squared_error(y, predictions)

    return best_features, mse


def backward_elimination(X, y, criterion="AIC"):
    best_features = X.columns.tolist()

    X_model = sm.add_constant(X[best_features])
    model = sm.OLS(y, X_model).fit()
    current_score = model.aic if criterion == "AIC" else model.bic

    while len(best_features) > 0:
        scores_with_candidates = []
        for candidate in best_features:
            features = best_features.copy()
            features.remove(candidate)
            X_model = sm.add_constant(X[features])
            model = sm.OLS(y, X_model).fit()
            score = model.aic if criterion == "AIC" else model.bic
            scores_with_candidates.append((score, candidate))

        # candidate whose removal gives the lowest score
        scores_with_candidates.sort()
        best_new_score, worst_candidate = scores_with_candidates[0]

        if best_new_score < current_score:
            best_features.remove(worst_candidate)
            current_score = best_new_score
        else:
            break

    beat_featured_model = sm.OLS(y, sm.add_constant(X[best_features])).fit()
    predictions = beat_featured_model.predict(sm.add_constant(X[best_features]))
    mse = mean_squared_error(y, predictions)
    return best_features, mse
