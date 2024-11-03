from ubiquitous_computing_machine.utils import load_data2

file = r"ubiquitous_computing_machine\data\pimaindiansdiabetes2.csv"

X, y = load_data2(file, header=1)  # (768, 8) (768,) panda data frame

print(X.shape, y.shape)
print(X.head())
import pandas as pd
from sklearn.impute import SimpleImputer

# Impute missing values with median
imputer = SimpleImputer(strategy="median")
data = imputer.fit_transform(X)
data = pd.DataFrame(
    data,
    columns=[
        "pregnant",
        "glucose",
        "pressure",
        "triceps",
        "insulin",
        "mass",
        "pedigree",
        "age",
    ],
)

# Add y as the last column with name 'diabetes'
data['diabetes'] = y

data.to_csv(
    r"ubiquitous_computing_machine\data\pimaindiansdiabetes2_imputed.csv", index=False
)
