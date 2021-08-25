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

    # Note: predictors x3, x4, x5, x6, ... are noise (their true coefficients are zero)
    for idx in range(3, 10):
        colname = f"x{idx}"
        df[colname] = np.random.uniform(size=n_obs) + np.random.uniform(size=n_obs)

    df["epsilon"] = np.random.normal(size=n_obs)

    # Note: this is the true regression equation
    df["y"] = 5 + true_beta0 * df["x0"] + 2 * df["x1"] - 1 * df["x2"] + df["epsilon"]

    return df


def run_regression_v0(df_train):

    model = LinearRegression()

    X = df_train[["x0", "x1", "x2"]]
    y = df_train["y"]

    model.fit(X=X, y=y)

    estimated_coefficients = model.coef_

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

    ci_lower = estimated_coefficients - 1.96 * std_errors
    ci_upper = estimated_coefficients + 1.96 * std_errors

    # TODO Also return sigma_squared_hat?
    return estimated_coefficients, ci_lower, ci_upper


def run_regression_v1(df_train):

    # TODO Keep adding predictors until beta_1 is statistically significantly < 1

    return estimated_coefficients, ci_lower, ci_upper


def main():

    np.random.seed(1234321)

    true_beta0 = 1.5

    n_replications = 300
    coef_replications = []
    ci_contains_beta0_replications = []

    for _ in range(n_replications):

        df = simulate_df(true_beta0=true_beta0)

        coefs, ci_lower, ci_upper = run_regression_v0(df)
        coef_replications.append(coefs)

        ci_contains_beta0 = ci_lower[0] < true_beta0 < ci_upper[0]
        ci_contains_beta0_replications.append(ci_contains_beta0)

    ci_coverage = np.mean(ci_contains_beta0_replications)
    print(f"Coverage of 95% confidence interval for beta_0: {ci_coverage}")

    print(pd.DataFrame(coef_replications))
    print(pd.DataFrame(coef_replications).describe())

    # TODO Plot histogram showing (a) true coefficient value, (b) distribution of coef estimates

    # TODO Another one with publication bias, drawer


if __name__ == "__main__":
    main()
