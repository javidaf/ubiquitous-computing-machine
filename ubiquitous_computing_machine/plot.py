import numpy as np
from ubiquitous_computing_machine.utils import load_data
import matplotlib.pyplot as plt


def plot_two_features(X, f1, f2):
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

    feature1 = X.iloc[:, f1]
    feature2 = X.iloc[:, f2]

    plt.figure(figsize=(10, 6))
    plt.scatter(feature1, feature2, alpha=0.5)
    plt.title("Scatter plot of two features")
    plt.xlabel(f"{feature_mapping[f1]}")
    plt.ylabel(f"{feature_mapping[f2]}")
    plt.grid(True)
    plt.show()


def plot_model_comparison(
    train_error,
    test_error,
    r2_train,
    r2_test,
    train_error_ii,
    test_error_ii,
    r2_train_ii,
    r2_test_ii,
):
    labels = ["Linear Effect", "Dummy Encoding"]
    mse_train_values = [train_error, train_error_ii]
    mse_test_values = [test_error, test_error_ii]
    r2_train_values = [r2_train, r2_train_ii]
    r2_test_values = [r2_test, r2_test_ii]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot
    rects1 = ax[0].bar(x - width / 2, mse_train_values, width, label="Train MSE")
    rects2 = ax[0].bar(x + width / 2, mse_test_values, width, label="Test MSE")

    ax[0].set_ylabel("MSE")
    ax[0].set_title("Mean Squared Error by Model Type")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].legend()

    # Plot
    rects3 = ax[1].bar(x - width / 2, r2_train_values, width, label="Train R^2")
    rects4 = ax[1].bar(x + width / 2, r2_test_values, width, label="Test R^2")

    ax[1].set_ylabel("R^2 Score")
    ax[1].set_title("R^2 Score by Model Type")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].legend()

    fig.tight_layout()

    plt.show()


# --------------------------------------------------
# --------------------------------------------------
##plot for part_1b.py


def plot_distributions(linear, dummy, label):
    plt.figure(figsize=(10, 4))
    plt.hist(linear, bins=40, alpha=0.5, label="Linear Effect", color="blue")
    plt.hist(dummy, bins=40, alpha=0.5, label="Dummy Encoding", color="orange")
    plt.xlabel(f"{label}")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {label} over 200 Runs")
    plt.legend()
    plt.show()

    # Calculate and print average test errors
    avg_mse_i = np.mean(linear)
    avg_mse_ii = np.mean(dummy)
    print(f"Average {label} - Linear Effect: {avg_mse_i:.4f}")
    print(f"Average {label} - Dummy Encoding: {avg_mse_ii:.4f}")
