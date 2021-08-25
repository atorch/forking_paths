import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simulate_df(true_beta0, n_obs=500):

    df = pd.DataFrame(
        {
            "x0": np.random.uniform(size=n_obs),
            "x1": np.random.uniform(size=n_obs) + np.random.uniform(size=n_obs),
        }
    )

    df["x2"] = df["x0"] + df["x1"] + np.random.normal(size=n_obs)

    # Note: predictors x3, x4, x5, x6, ... have true coefficients equal to zero
    # (they have no effect on y), but they are correlated with x0
    for idx in range(3, 25):
        colname = f"x{idx}"
        df[colname] = (
            df["x0"] + np.random.uniform(size=n_obs) + np.random.uniform(size=n_obs)
        )

    df["epsilon"] = np.random.normal(size=n_obs)

    # Note: this is the true regression equation
    df["y"] = 5 + true_beta0 * df["x0"] + 2 * df["x1"] - 1 * df["x2"] + df["epsilon"]

    return df


def calculate_std_errors(model, y, X):

    residuals = y - model.predict(X)

    # Note: we need to include the constant when calculating (X^T X)^-1
    X_with_constant = np.hstack([np.ones((X.shape[0], 1)), X])

    X_transpose_X_inverse = np.linalg.inv(
        np.matmul(np.transpose(X_with_constant), X_with_constant)
    )

    # Note the degrees of freedom correction (accounting for the constant term)
    # See https://web.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf
    sigma_squared_hat = np.sum(residuals ** 2) / (X.shape[0] - X_with_constant.shape[1])

    std_errors = np.sqrt(sigma_squared_hat * np.diag(X_transpose_X_inverse)[1:])

    return std_errors


def run_regression_v0(df_train):

    """
    This function mimics a researcher doing 'the right thing':
    they decided ahead of time that they'll run a regression of
    y on x0, x1 and x2 (and a constant). They run one regression, report
    their coefficient estimates and confidence intervals, and they're done.
    """

    model = LinearRegression()

    X = df_train[["x0", "x1", "x2"]]
    y = df_train["y"]

    model.fit(X=X, y=y)

    # TODO Also return sigma_squared_hat?
    std_errors = calculate_std_errors(model, y, X)

    estimated_coefficients = model.coef_
    ci_lower = estimated_coefficients - 1.96 * std_errors
    ci_upper = estimated_coefficients + 1.96 * std_errors

    return estimated_coefficients, ci_lower, ci_upper


def run_regression_v1(df_train, beta0_threshold=1.75, max_iterations=20):

    """
    This function mimics a researcher following forking paths:
    they start with a regression of
    y on x0, x1 and x2 (and a constant), but then examine the estimated
    coefficient for beta0, check whether it is larger than beta0_threshold,
    and (if so) keep adding additional predictors and rerunning the regression
    (for a maximum of max_iterations steps). The analysis -- specifically the set of regression
    coefficients -- varies based on the data.
    """

    for last_predictor_idx in range(2, 2 + max_iterations + 1):

        predictors = [f"x{idx}" for idx in range(0, last_predictor_idx + 1)]
        # print(f"Running regression with predictors {predictors}")

        model = LinearRegression()

        X = df_train[predictors]
        y = df_train["y"]

        model.fit(X=X, y=y)

        estimated_coefficients = model.coef_

        # Imagine that the researcher has some reason (theory, chances of publication, preconception, etc)
        # to "want" their estimated beta0 to be larger than beta0_threshold,
        # or to think that something is wrong with their analysis if their estimate is below the threshold.
        # They may not be actively cheating, but instead convince themselves that their regression
        # failed to control for some confounder (which will be added to the set of predictors on the next iteration)
        if estimated_coefficients[0] > beta0_threshold:
            break

    # print(f"Stopping forking paths at last_predictor_idx {last_predictor_idx}")
    std_errors = calculate_std_errors(model, y, X)

    ci_lower = estimated_coefficients - 1.96 * std_errors
    ci_upper = estimated_coefficients + 1.96 * std_errors

    # Only pay attention to estimated coefficients for beta0, beta1, beta2,
    # even though the model might include additional predictors
    return estimated_coefficients[0:3], ci_lower[0:3], ci_upper[0:3]


def main():

    np.random.seed(1234321)

    true_beta0 = 1.5

    for run_regression in [run_regression_v0, run_regression_v1]:

        n_replications = 500
        coef_replications = []
        ci_contains_beta0_replications = []

        for _ in range(n_replications):

            df_train = simulate_df(true_beta0=true_beta0)

            coefs, ci_lower, ci_upper = run_regression(df_train)
            coef_replications.append(coefs)

            ci_contains_beta0 = ci_lower[0] < true_beta0 < ci_upper[0]
            ci_contains_beta0_replications.append(ci_contains_beta0)

        print(f"*** results for {run_regression.__name__} ***")
        ci_coverage = np.mean(ci_contains_beta0_replications)
        print(f"Coverage of 95% confidence interval for beta_0: {ci_coverage}")

        df = pd.DataFrame(coef_replications, columns=["beta0", "beta1", "beta2"])
        print(df.describe())

        f, ax = plt.subplots(figsize=(12, 8))
        ax = df["beta0"].plot.hist(bins=50)
        plt.savefig(
            f"sampling_distribution_for_estimated_beta0_{run_regression.__name__}.png"
        )
        plt.close()

    # TODO Plot histogram showing (a) true coefficient value, (b) distribution of coef estimates

    # TODO Another one with publication bias, drawer


if __name__ == "__main__":
    main()
