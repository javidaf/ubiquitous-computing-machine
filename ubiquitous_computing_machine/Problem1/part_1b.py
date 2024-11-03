from matplotlib import pyplot as plt
import pandas as pd
from ubiquitous_computing_machine.Problem1.part_1a import (
    linear_effect,
    dummy_encoding_effect,
)
import seaborn as sns


def repeat_experiment(X, y, n_runs=200):
    train_errors_i = []
    train_errors_ii = []
    train_scores_i = []
    train_scores_ii = []

    test_errors_i = []
    test_errors_ii = []
    test_scores_i = []
    test_scores_ii = []

    betas_i = []
    betas_ii = []

    for _ in range(n_runs):

        # Option (i): Linear Effect
        eval_metrics_i, beta_i = linear_effect(X, y)
        mse_train, mse_test, r2_train, r2_test = (
            eval_metrics_i["train_error"],
            eval_metrics_i["test_error"],
            eval_metrics_i["r2_train"],
            eval_metrics_i["r2_test"],
        )
        test_errors_i.append(mse_test)
        test_scores_i.append(r2_test)
        train_errors_i.append(mse_train)
        train_scores_i.append(r2_train)
        betas_i.append(beta_i)

        # Option (ii): Dummy Encoding
        eval_metrics_ii, beta_ii = dummy_encoding_effect(X, y)
        train_error, test_error, r2_train, r2_test = (
            eval_metrics_ii["train_error"],
            eval_metrics_ii["test_error"],
            eval_metrics_ii["r2_train"],
            eval_metrics_ii["r2_test"],
        )
        test_errors_ii.append(test_error)
        test_scores_ii.append(r2_test)
        train_errors_ii.append(train_error)
        train_scores_ii.append(r2_train)
        betas_ii.append(beta_ii)

    eval_metrics = {
        "train_errors_i": train_errors_i,
        "train_errors_ii": train_errors_ii,
        "train_scores_i": train_scores_i,
        "train_scores_ii": train_scores_ii,
        "test_errors_i": test_errors_i,
        "test_errors_ii": test_errors_ii,
        "test_scores_i": test_scores_i,
        "test_scores_ii": test_scores_ii,
        "betas_i": betas_i,
        "betas_ii": betas_ii,
    }

    return eval_metrics


def visualize_coefficients(eval_metrics):
    feature_names = ["TPSA", "SAacc", "H050", "MLOGP", "RDCHI", "GATS1p", "nN", "C040"]

    df_i = pd.DataFrame(eval_metrics["betas_i"], columns=feature_names)
    df_ii = pd.DataFrame(eval_metrics["betas_ii"], columns=feature_names)

    df_i["Model"] = "Linear Effect"
    df_ii["Model"] = "Dummy Encoding"

    df_combined = pd.concat([df_i, df_ii], ignore_index=True)

    df_melted = df_combined.melt(
        id_vars="Model", var_name="Feature", value_name="Coefficient"
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    sns.boxplot(
        x="Coefficient",
        y="Feature",
        data=df_melted[df_melted["Model"] == "Linear Effect"],
        ax=axes[0],
    )
    axes[0].set_title("Linear Effect Coefficients")
    axes[0].set_xlabel("Coefficient Value")
    axes[0].set_ylabel("Feature")

    sns.boxplot(
        x="Coefficient",
        y="Feature",
        data=df_melted[df_melted["Model"] == "Dummy Encoding"],
        ax=axes[1],
    )
    axes[1].set_title("Dummy Encoding Coefficients")
    axes[1].set_xlabel("Coefficient Value")
    axes[1].set_ylabel("Feature")

    plt.tight_layout()
    plt.show()
