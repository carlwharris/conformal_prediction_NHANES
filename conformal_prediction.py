import numpy as np
from sklearn.linear_model import QuantileRegressor
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from preprocess_data import recode_variables, split_train_cal_test
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import d2_pinball_score, make_scorer
from qrnn import get_model, qloss, compute_quantile_loss, get_model_nn, qloss_nn
from keras.callbacks import *
import tensorflow as tf
import warnings

def mqloss(y_true, y_pred, alpha):
  if (alpha > 0) and (alpha < 1):
    residual = y_true - y_pred
    return np.mean(residual * (alpha - (residual<0)))
  else:
    return np.nan

class CP:
    def __init__(self, X_train, X_cal, y_train, y_cal, alpha, regressor="QuantileRegressor", verbose=1):
        self.X_train = X_train
        self.X_cal = X_cal
        self.y_train = y_train
        self.y_cal = y_cal
        self.alpha = alpha

        self.regressor = regressor
        self.verbose = verbose

        self.hyperparameters = None
        self.models = None


    def hyperparam_search(self, search_type="random", n_iter=50):

        self.hyperparameters = {}

        quantiles = [self.alpha / 2, 0.5, 1 - self.alpha / 2]
        labels = ['lower', 'median', 'upper']

        for label, quantile in zip(labels, quantiles):
            if self.regressor == "QuantileRegressor":
                reg = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
                alpha_vals = np.logspace(0.0, 1, 10, base=10) / 10. - 0.1
                params = [{'alpha': alpha_vals}]

            if self.regressor == "GradientBoostingRegressor":
                reg = GradientBoostingRegressor(alpha=quantile, loss='quantile')
                params = [{'learning_rate': [0.01, 0.1, 1],
                           'n_estimators': [10, 50, 100, 100],
                           'max_depth': [1, 3, 5, 10, 25, 50],
                           'subsample': [.5, .75, 1]}]

            if search_type == "random":
                searcher = RandomizedSearchCV(reg,
                                              param_distributions=params,
                                              scoring=make_scorer(d2_pinball_score, alpha=quantile),
                                              cv=5, verbose=self.verbose, n_jobs=-1, n_iter=n_iter)

            if search_type == "grid":
                searcher = GridSearchCV(reg,
                                  param_grid=params,
                                  scoring=make_scorer(d2_pinball_score, alpha=quantile),
                                  cv=5, verbose=self.verbose)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                searcher.fit(self.X_train, self.y_train)
            print(searcher.best_score_)
            print(searcher.best_params_)

            self.hyperparameters[label] = searcher.best_params_

    # Train a lower (qr_lower) and upper (qr_upper) quantile regressor, as well
    # as a median (qr_med) regressor
    def train(self):
        quantiles = [self.alpha / 2, 0.5, 1 - self.alpha / 2]
        labels = ['lower', 'median', 'upper']

        self.models = {}
        for label, quantile in zip(labels, quantiles):
            if self.verbose == 1:
                print(f"Fitting {label} quantile ({quantile})")

            if self.regressor == "nn":
                input_dim = self.X_train.shape[1]
                num_hidden_layers = 1
                num_units = [200]
                act = ['relu']
                gauss_std = [0]
                num_quantiles = 39

                # Get model
                model = get_model_nn(input_dim, num_units, act, gauss_std=gauss_std,
                                     num_hidden_layers=num_hidden_layers)

                early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                model.compile(loss=lambda y_t, y_p: qloss_nn(y_true=y_t, y_pred=y_p, q=quantile),
                              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
                # model.compile(loss=lambda y_t, y_p: compute_quantile_loss(y_true=y_t, y_pred=y_p, quantile=0.05), optimizer='adam')

                model.fit(x=self.X_train.astype('float32'), y=self.y_train.astype('float32'),
                          epochs=100,
                          validation_split=0.25,
                          batch_size=16,
                          shuffle=True,
                          callbacks=[early_stopping])

            if self.regressor == "QuantileRegressor":
                model = QuantileRegressor(quantile=quantile, solver="highs", **self.hyperparameters['lower'])
                model.fit(self.X_train, self.y_train)

            if self.regressor == "GradientBoostingRegressor":
                model = GradientBoostingRegressor(alpha=quantile, loss='quantile', **self.hyperparameters['lower'])
                model.fit(self.X_train, self.y_train)

            self.models[label] = model

    # Given df X, predict lower, upper, and median quantiles using regressors. Return
    # as dataframe with columns y_median, y_lower, y_upper
    def predict_quantiles(self, X, y_true):
        df_dict = {'y_true': y_true}
        for label in ['lower', 'median', 'upper']:
            curr_preds = self.models[label].predict(X.astype('float32'))
            print(curr_preds.shape)
            df_dict[label] = curr_preds.flatten()
        pred_df = pd.DataFrame(df_dict, index=X.index)
        return pred_df

    # Calculate scores using X_cal and lower/upper quantiles. Return scores and
    # save to object as self.scores
    def calculate_scores(self):
        y_df = self.predict_quantiles(self.X_cal, self.y_cal)
        lower_diff = y_df['lower'] - self.y_cal
        upper_diff = self.y_cal - y_df['upper']
        scores = pd.concat([lower_diff, upper_diff], axis=1).max(axis=1)
        scores.name = 'scores'
        self.scores = scores
        return scores

    # Calculate the quantile of the score at level alpha, return qhat and save to object
    def calc_qhat(self):
        n = len(self.scores)
        qhat = np.quantile(self.scores, np.ceil((n + 1) * (1 - self.alpha)) / n, method='higher')
        self.qhat = qhat

        if self.verbose == 1:
            print(f"At alpha={self.alpha}, qhat = {qhat}")
        return qhat

    # Given dataframe with entries y_lower and y_upper, scale by q_hat and return
    # df with (calibrated) entries y_lower and y_upper
    def conformalize_CIs(self, pred_df, qhat=None):
        if qhat == None:
            qhat = self.qhat

        cp_pred_df = pred_df.copy()

        if self.verbose == 1:
            print(f"Conformalizing with qhat={qhat}")

        cp_pred_df['lower'] = cp_pred_df['lower'] - qhat
        cp_pred_df['upper'] = cp_pred_df['upper'] + qhat
        cp_pred_df['diff'] = cp_pred_df['upper'] - cp_pred_df['lower']
        return cp_pred_df

# Given dataframe with estimated lower and upper bounds, and the true value, calculate
# the proportion of rows where the true value falls wtihin the predicted confidence
# interval
def calc_coverage(cp_df):
    greq_lb = cp_df['y_true'] >= cp_df['lower']
    leq_ub = cp_df['y_true'] <= cp_df['upper']
    is_in_interval = greq_lb & leq_ub
    return np.mean(is_in_interval)

def run_CP_pipeline(X, y, alpha=0.05):
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_train_cal_test(X, y, trn_prop=0.5, cal_prop=0.25)

    reg = CP(X_train, X_cal, y_train, y_cal)
    reg.train(alpha=alpha, regressor="QuantileRegressor")
    reg.calculate_scores()
    reg.calc_qhat(alpha=alpha)
    test_pred_df = reg.predict_quantiles(X_test, y_test = y_test)

    test_cp_df = reg.conformalize_CIs(test_pred_df)

    return test_cp_df, X_test
