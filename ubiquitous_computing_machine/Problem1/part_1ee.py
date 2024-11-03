from statsmodels.gam.api import GLMGam, BSplines
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from ubiquitous_computing_machine.utils import get_data

X, y = get_data()

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


combined_min = np.minimum(X_train.min(), X_test.min())
combined_max = np.maximum(X_train.max(), X_test.max())

# Extend the knots beyond the combined data range
margin = 0.1  # 10% margin

knot_kwds = []
for col in X_train.columns:
    data_min = combined_min[col]
    data_max = combined_max[col]
    data_range = data_max - data_min
    knot_min = data_min - margin * data_range
    knot_max = data_max + margin * data_range
    knot_kwds.append({"lower_bound": knot_min, "upper_bound": knot_max})


spline_basis = BSplines(
    X_train,
    df=[4] * X_train.shape[1],
    degree=[3] * X_train.shape[1],
    variable_names=X_train.columns.tolist(),
    knot_kwds=knot_kwds,
)


gam = GLMGam(y_train, smoother=spline_basis).fit()


y_train_pred = gam.predict(exog=None, exog_smooth=X_train)
y_test_pred = gam.predict(exog=None, exog_smooth=X_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("GAM - Training MSE:", mse_train)
print("GAM - Test MSE:", mse_test)

fig, axs = plt.subplots(4, 2, figsize=(15, 20))
for i, ax in enumerate(axs.flatten()):
    gam.plot_partial(i, ax=ax)
    ax.set_title(X.columns[i])
plt.tight_layout()
plt.show()
