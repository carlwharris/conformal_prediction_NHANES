import numpy as np
from sklearn.linear_model import QuantileRegressor
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from preprocess_data import  split_train_cal_test
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import d2_pinball_score, make_scorer

from qrnn import create_nn_model, qloss_nn
from keras.callbacks import *
import tensorflow as tf
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from qrnn import NN_Estimator

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class CP:
    # regressor: QuantileRegressor/QR, GradientBoostingRegressor/QBR, NN
    def __init__(self, X_train, X_cal, y_train, y_cal, alpha, regressor="QR", verbose=1):
        if regressor == "QR":
            regressor = "QuantileRegressor"
        elif regressor == "GBR":
            regressor = "GradientBoostingRegressor"

        self.X_train = X_train
        self.X_cal = X_cal
        self.y_train = y_train
        self.y_cal = y_cal
        self.alpha = alpha

        self.regressor = regressor
        self.verbose = verbose

        self.hyperparameters = None
        self.models = None
        self.scores = None

    # search_type: RandomizedSearchCV ("random") or  GridSearchCV("grid")
    def hyperparam_search(self, search_type="random", n_iter=50):
        self.hyperparameters = {}
        quantiles = [self.alpha / 2, 0.5, 1 - self.alpha / 2]
        labels = ['lower', 'median', 'upper']

        for label, quantile in zip(labels, quantiles):
            mdl = self._get_untrained_mdl(label)
            params = self._get_hyperparameter_set()

            if search_type == "random":
                searcher = RandomizedSearchCV(mdl, param_distributions=params,
                                              scoring=make_scorer(d2_pinball_score, alpha=quantile),
                                              cv=5, verbose=self.verbose, n_jobs=-1, n_iter=n_iter)

            if search_type == "grid":
                searcher = GridSearchCV(mdl, param_grid=params, scoring=make_scorer(d2_pinball_score, alpha=quantile),
                                        cv=5, verbose=self.verbose)

            self.txt_out(f"Hyperparameter optimization for {self.regressor} model using {search_type} search over parameter set:")
            self.txt_out(params)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                searcher.fit(self.X_train, self.y_train)

            self.txt_out(f"  - best score: {searcher.best_score_}")
            self.txt_out(f"  - best params: {searcher.best_params_}\n")

            self.hyperparameters[label] = searcher.best_params_

    def _get_hyperparameter_set(self):
        if self.regressor == "QuantileRegressor":
            alpha_vals = np.logspace(0.0, 1, 10, base=10) / 10. - 0.1
            params = [{'alpha': alpha_vals}]

        if self.regressor == "GradientBoostingRegressor":
            params = [{'learning_rate': [0.01, 0.1, 1],
                       'n_estimators': [10, 50, 100, 100],
                       'max_depth': [1, 3, 5, 10, 25, 50],
                       'subsample': [.5, .75, 1]}]

        if self.regressor == "NN":
            params = [{'n_units': [10, 20, 50, 100],
                       'n_layers': [1, 2, 3],
                       'alpha': [self.alpha]}]
        return params

    def _get_untrained_mdl(self, label):
        labels = ['lower', 'median', 'upper']
        quantiles = [self.alpha / 2, 0.5, 1 - self.alpha / 2]

        if self.hyperparameters is None:
            hyperparams = {}
        else:
            if label in self.hyperparameters.keys():
                hyperparams = self.hyperparameters[label]
            else:
                hyperparams = {}

        alpha = quantiles[labels.index(label)]
        if self.regressor == "QuantileRegressor":
            if len(hyperparams) == 0:
                hyperparams = {'alpha': 0}
            model = QuantileRegressor(quantile=alpha, solver="highs", **hyperparams)

        if self.regressor == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(alpha=alpha, loss='quantile', **hyperparams)

        if self.regressor == "NN":
            model = NN_Estimator(**hyperparams)

        return model

    # Train a lower (qr_lower) and upper (qr_upper) quantile regressor, as well
    # as a median (qr_med) regressor
    def train(self):
        quantiles = [self.alpha / 2, 0.5, 1 - self.alpha / 2]
        labels = ['lower', 'median', 'upper']

        self.models = {}
        for label, quantile in zip(labels, quantiles):
            if self.verbose == 1:
                print(f"Fitting {label} quantile ({quantile})")

            model = self._get_untrained_mdl(label)
            if self.regressor == "QuantileRegressor":
                model.fit(self.X_train, self.y_train)

            if self.regressor == "GradientBoostingRegressor":
                model.fit(self.X_train, self.y_train)

            if self.regressor == "NN":
                model.fit(self.X_train, self.y_train)

            self.models[label] = model

    # Calculate the quantile of the score at level alpha, return qhat and save to object
    def calc_qhat(self):
        if self.scores is None:
            self.calculate_scores()

        n = len(self.scores)
        qhat = np.quantile(self.scores, np.ceil((n + 1) * (1 - self.alpha)) / n, method='higher')
        self.qhat = qhat

        if self.verbose == 1:
            print(f"At alpha={self.alpha}, qhat = {qhat}")
        return qhat

    # Calculate scores using X_cal and lower/upper quantiles. Return scores and
    # save to object as self.scores
    def calculate_scores(self):
        if self.verbose == 1:
            print("Calculating scores")

        y_df = self.predict_mdl_quantiles(self.X_cal, self.y_cal)
        lower_diff = y_df['lower'] - self.y_cal
        upper_diff = self.y_cal - y_df['upper']
        scores = pd.concat([lower_diff, upper_diff], axis=1).max(axis=1)
        scores.name = 'scores'
        self.scores = scores
        return scores

    def predict_cp_quantiles(self, X, y_true=None):
        init_preds = self.predict_mdl_quantiles(X, y_true)
        cp_pred_df = self.conformalize_CIs(init_preds)
        return cp_pred_df

    # Given df X, predict lower, upper, and median quantiles using regressors. Return
    # as dataframe with columns y_median, y_lower, y_upper
    def predict_mdl_quantiles(self, X, y_true=None):
        if y_true is None:
            df_dict = {}
        else:
            df_dict = {'y_true': y_true}

        for label in ['lower', 'median', 'upper']:
            curr_preds = self.models[label].predict(X.astype('float32'))
            df_dict[label] = curr_preds.flatten()
        pred_df = pd.DataFrame(df_dict, index=X.index)
        return pred_df

    # Given dataframe with entries y_lower and y_upper, scale by q_hat and return
    # df with (calibrated) entries y_lower and y_upper
    def conformalize_CIs(self, pred_df, qhat=None):
        if qhat is None:
            qhat = self.qhat

        cp_pred_df = pred_df.copy()

        if self.verbose == 1:
            print(f"Conformalizing with qhat={qhat}")

        cp_pred_df['lower'] = cp_pred_df['lower'] - qhat
        cp_pred_df['upper'] = cp_pred_df['upper'] + qhat
        cp_pred_df['diff'] = cp_pred_df['upper'] - cp_pred_df['lower']
        return cp_pred_df

    def txt_out(self, s, lvl=1):
        if self.verbose == 0:
            return

        if self.verbose >= lvl:
            print(s)

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
