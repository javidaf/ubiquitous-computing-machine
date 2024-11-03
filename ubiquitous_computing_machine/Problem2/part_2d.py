import numpy as np
from ubiquitous_computing_machine.utils import load_data2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


file = r"ubiquitous_computing_machine\data\pimaindiansdiabetes2_imputed.csv"
X, y = load_data2(file, header=1)  # (768, 8) (768,) panda data frame


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=42, stratify=y
)

NN = MLPClassifier(
    max_iter=800,
    activation="logistic",
    random_state=42,
    hidden_layer_sizes=(500,),
    learning_rate="adaptive",

)


param_grid = {
    "alpha": np.logspace(-4.5, 1, 100),
}


grid_search = GridSearchCV(NN, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


print("Best parameters: ", grid_search.best_params_)



print("Best score: ", 1 - grid_search.best_score_)
