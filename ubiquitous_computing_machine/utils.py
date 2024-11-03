import pandas as pd

file = r"data\qsar_aquatic_toxicity.csv"
file2 = r"data\pimaindiansdiabetes2_imputed.csv"
feature_names = [
    "TPSA(Tot)",
    "SAacc",
    "H-050",
    "MLOGP",
    "RDCHI",
    "GATS1p",
    "nN",
    "C-040",
]

feature_mapping = {
    0: "TPSA(Tot)",
    1: "SAacc",
    2: "H-050",  # descrete values
    3: "MLOGP",
    4: "RDCHI",
    5: "GATS1p",
    6: "nN",  # descrete values
    7: "C-040",  # descrete values
}


def load_data(file_path, header=None):
    data = pd.read_csv(file_path, delimiter=";", header=header)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def load_data2(file_path, header=None):
    data = pd.read_csv(file_path, delimiter=",", header=0)
    X = data.iloc[:, :8]
    y = data.iloc[:, -1]
    return X, y


def get_data():
    return load_data(file)


def get_data2():
    return load_data2(file2)


def dummy_encode(X):
    return (X > 0).astype(int)


feature_names_p2 = [
    "pregnant",
    "glucose",
    "pressure",
    "triceps",
    "insulin",
    "mass",
    "pedigree",
    "age",
    "diabetes",  # Target
]

feature_mapping_p2 = {
    0: "pregnant",
    1: "glucose",
    2: "pressure",
    3: "triceps",
    4: "insulin",
    5: "mass",
    6: "pedigree",
    7: "age",
    8: "diabetes",
}

## Downloading the dataset
# import requests

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv"
# response = requests.get(url)

# with open("qsar_aquatic_toxicity.csv", "wb") as file:
#     file.write(response.content)

# print("File downloaded successfully.")
