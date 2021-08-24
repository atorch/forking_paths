import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simulate_df(true_beta0, n_obs=500):

    df = pd.DataFrame({
        "x0": np.random.uniform(size=n_obs),
        "x1": np.random.uniform(size=n_obs) + np.random.uniform(size=n_obs),
    })

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

    X_transpose_X_inverse = np.linalg.inv(np.matmul(np.transpose(X), X))

    # Note the degrees of freedom correction
    sigma_squared_hat = np.sum(residuals ** 2) / (X.shape[0] - X.shape[1])

    std_errors = np.sqrt(sigma_squared_hat) * np.diag(X_transpose_X_inverse)

    ci_lower = estimated_coefficients - 1.96 * std_errors
    ci_upper = estimated_coefficients + 1.96 * std_errors

    return estimated_coefficients, ci_lower, ci_upper


def run_regression_v1(df_train):

    # TODO Keep adding predictors until beta_1 is statistically significantly < 1

    return estimated_coefficients, ci_lower, ci_upper


def main():

    np.random.seed(1234321)

    true_beta0 = 1.5

    df = simulate_df(true_beta0=true_beta0)

    coefs, ci_lower, ci_upper = run_regression_v0(df)

    ci_contains_beta0 = ci_lower[0] < true_beta0 < ci_upper[0]

    # TODO CI coverage
    # TODO Plot histogram showing (a) true coefficient value, (b) distribution of coef estimates


if __name__ == "__main__":
    main()
