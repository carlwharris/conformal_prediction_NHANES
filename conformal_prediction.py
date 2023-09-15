import numpy as np
from sklearn.linear_model import QuantileRegressor
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

class CP:
    def __init__(self, X_train, X_cal, y_train, y_cal, verbose=1):
        self.X_train = X_train
        self.X_cal = X_cal
        self.y_train = y_train
        self.y_cal = y_cal
        self.verbose = verbose


    # Train a lower (qr_lower) and upper (qr_upper) quantile regressor, as well
    # as a median (qr_med) regressor
    def train(self, alpha, regressor="QuantileRegressor"):
        self.alpha = alpha

        if regressor == "QuantileRegressor":
            qr_lower = QuantileRegressor(quantile=alpha / 2, alpha=0, solver="highs")

            if self.verbose == 1:
                print(f"Fitting lower quantile ({alpha / 2})")
            qr_lower.fit(self.X_train, self.y_train)

            qr_upper = QuantileRegressor(quantile=1 - alpha / 2, alpha=0, solver="highs")
            if self.verbose == 1:
                print(f"Fitting upper quantile ({1 - alpha / 2})")
            qr_upper.fit(self.X_train, self.y_train)

            qr_med = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")
            if self.verbose == 1:
                print(f"Fitting median quantile (0.5)")
            qr_med.fit(self.X_train, self.y_train)

        if regressor == "GradientBoostingRegressor":
            qr_lower = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=100, alpha=1-alpha/2, loss='quantile')

            if self.verbose == 1:
                print(f"Fitting lower quantile ({alpha / 2})")
            qr_lower.fit(self.X_train, self.y_train)

            qr_upper = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=100, alpha=alpha/2, loss='quantile')
            if self.verbose == 1:
                print(f"Fitting upper quantile ({1 - alpha / 2})")
            qr_upper.fit(self.X_train, self.y_train)

            qr_med = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=100)
            if self.verbose == 1:
                print(f"Fitting median quantile (0.5)")
            qr_med.fit(self.X_train, self.y_train)

        self.qr_lower = qr_lower
        self.qr_upper = qr_upper
        self.qr_med = qr_med

    # Given df X, predict lower, upper, and median quantiles using regressors. Return
    # as dataframe with columns y_median, y_lower, y_upper
    def predict_quantiles(self, X, y_test=None):
        y_lower = self.qr_lower.predict(X)
        y_upper = self.qr_upper.predict(X)
        y_median = self.qr_med.predict(X)

        if y_test is None:
            pd_dict = {'y_median': y_median,
                       'y_lower': y_lower,
                       'y_upper': y_upper}
        else:
            pd_dict = {'y_true': y_test,
                        'y_median': y_median,
                       'y_lower': y_lower,
                       'y_upper': y_upper}
        df = pd.DataFrame(pd_dict)
        df.index = X.index
        return df

    # Calculate scores using X_cal and lower/upper quantiles. Return scores and
    # save to object as self.scores
    def calculate_scores(self):
        y_df = self.predict_quantiles(self.X_cal)
        lower_diff = y_df['y_lower'] - self.y_cal
        upper_diff = self.y_cal - y_df['y_upper']
        scores = pd.concat([lower_diff, upper_diff], axis=1).max(axis=1)
        scores.name = 'scores'
        self.scores = scores
        return scores

    # Calculate the quantile of the score at level alpha, return qhat and save to object
    def calc_qhat(self, alpha):
        n = len(self.scores)
        qhat = np.quantile(self.scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
        self.qhat = qhat

        if self.verbose == 1:
            print(f"At alpha={alpha}, qhat = {qhat}")
        return qhat

    # Given dataframe with entries y_lower and y_upper, scale by q_hat and return
    # df with (calibrated) entries y_lower and y_upper
    def conformalize_CIs(self, qr_df, qhat=None):
        if qhat == None:
            qhat = self.qhat

        ci_df = qr_df.copy()

        if self.verbose == 1:
            print(f"Conformalizing with qhat={qhat}")

        ci_df['y_lower'] = ci_df['y_lower'] - qhat
        ci_df['y_upper'] = ci_df['y_upper'] + qhat
        ci_df['diff'] = ci_df['y_upper'] - ci_df['y_lower']
        return ci_df

    # Given dataframe with estimated lower and upper bounds, and the true value, calculate
    # the proportion of rows where the true value falls wtihin the predicted confidence
    # interval
    def calc_coverage(self, cp_df):
        greq_lb = cp_df['y_true'] >= cp_df['y_lower']
        leq_ub = cp_df['y_true'] <= cp_df['y_upper']
        is_in_interval = greq_lb & leq_ub
        return np.mean(is_in_interval)






