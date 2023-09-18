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

def mqloss(y_true, y_pred, alpha):
  if (alpha > 0) and (alpha < 1):
    residual = y_true - y_pred
    return np.mean(residual * (alpha - (residual<0)))
  else:
    return np.nan

class CP:
    def __init__(self, X_train, X_cal, y_train, y_cal, verbose=1):
        self.X_train = X_train
        self.X_cal = X_cal
        self.y_train = y_train
        self.y_cal = y_cal
        self.verbose = verbose
        self.hyperparameters = None
        self.model = None

    def hyperparam_search(self, alpha, regressor="QuantileRegressor"):

        hyperparams = {}
        if regressor == "QuantileRegressor":
            qr_lower = QuantileRegressor(quantile=alpha / 2, alpha=0, solver="highs")
            alpha_vals = np.logspace(0.0, 1, 10, base=10) / 10. - 0.1
            params = [{'alpha': alpha_vals}]
            gs_knn = GridSearchCV(qr_lower,
                                  param_grid=params,
                                  scoring='r2',
                                  cv=5, verbose=self.verbose)
            gs_knn.fit(self.X_train, self.y_train)
            print(gs_knn.best_params_)

        if regressor == "GradientBoostingRegressor":
            for i, curr_alpha in enumerate([alpha/2, 1-alpha/2]):

                curr_reg = GradientBoostingRegressor(alpha=curr_alpha, loss='quantile')
                params = [{'learning_rate': [0.01, 0.1, 1],
                           'n_estimators': [10, 50, 100, 100],
                           'max_depth': [1, 3, 5, 10, 25, 50],
                           'subsample':[.5, .75, 1]}
                ]
                searcher = RandomizedSearchCV(curr_reg,
                                      param_distributions=params,
                                      scoring=make_scorer(d2_pinball_score, alpha=curr_alpha),
                                      # scoring=make_scorer(mqloss, alpha=curr_alpha),
                                      cv=5, verbose=1,
                                            n_jobs=-1,
                                            n_iter=100)
                # gs_knn = GridSearchCV(qr_lower,
                #                       param_grid=params,
                #                       scoring='r2',
                #                       cv=2, verbose=2,
                #                             n_jobs=-1)
                searcher.fit(self.X_train, self.y_train)
                print(searcher.best_params_)
                print(searcher.best_score_)

                if i == 0:
                    hyperparams['lower'] = searcher.best_params_
                else:
                    hyperparams['upper'] = searcher.best_params_

        self.hyperparameters = hyperparams

    # Train a lower (qr_lower) and upper (qr_upper) quantile regressor, as well
    # as a median (qr_med) regressor
    def train(self, alpha, regressor="QuantileRegressor"):
        self.alpha = alpha

        if regressor == "qrnn":
            # Parameters
            input_dim = 17
            num_hidden_layers = 2
            num_units = [50, 50]
            act = ['relu', 'relu']
            dropout = [0.1, 0.1]
            gauss_std = [0.3, 0.3]
            num_quantiles = 39

            # Get model
            model = get_model(input_dim, num_units, act, dropout, gauss_std, num_hidden_layers, num_quantiles)
            print(model.summary())

            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            model.compile(loss=lambda y_t, y_p: qloss(y_true=y_t, y_pred=y_p, n_q=num_quantiles),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
            # model.compile(loss=lambda y_t, y_p: compute_quantile_loss(y_true=y_t, y_pred=y_p, quantile=0.05), optimizer='adam')
            model.fit(x=self.X_train.astype('float32'), y=self.y_train.astype('float32'),
                      epochs=25,
                      validation_split=0.2,
                      batch_size=16,
                      shuffle=True,
                      callbacks=[early_stopping])
            self.model = model
            return model

        if regressor == "nn":
            # Parameters
            input_dim = 17
            num_hidden_layers = 1
            num_units = [200]
            act = ['relu', ]
            gauss_std = [0]
            num_quantiles = 39

            # Get model
            model = get_model_nn(input_dim, num_units, act, gauss_std=gauss_std, num_hidden_layers=num_hidden_layers)

            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            model.compile(loss=lambda y_t, y_p: qloss_nn(y_true=y_t, y_pred=y_p, q=0.025),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
            # model.compile(loss=lambda y_t, y_p: compute_quantile_loss(y_true=y_t, y_pred=y_p, quantile=0.05), optimizer='adam')
            model.fit(x=self.X_train.astype('float32'), y=self.y_train.astype('float32'),
                      epochs=100,
                      validation_split=0.25,
                      batch_size=16,
                      shuffle=True,
                      callbacks=[early_stopping])
            qr_lower = model

            model = get_model_nn(input_dim, num_units, act, gauss_std=gauss_std, num_hidden_layers=num_hidden_layers)

            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            model.compile(loss=lambda y_t, y_p: qloss_nn(y_true=y_t, y_pred=y_p, q=0.5),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
            # model.compile(loss=lambda y_t, y_p: compute_quantile_loss(y_true=y_t, y_pred=y_p, quantile=0.05), optimizer='adam')
            model.fit(x=self.X_train.astype('float32'), y=self.y_train.astype('float32'),
                      epochs=100,
                      validation_split=0.25,
                      batch_size=16,
                      shuffle=True,
                      callbacks=[early_stopping])
            qr_med = model


            model = get_model_nn(input_dim, num_units, act, gauss_std=gauss_std, num_hidden_layers=num_hidden_layers)
            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            model.compile(loss=lambda y_t, y_p: qloss_nn(y_true=y_t, y_pred=y_p, q=0.975),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
            # model.compile(loss=lambda y_t, y_p: compute_quantile_loss(y_true=y_t, y_pred=y_p, quantile=0.05), optimizer='adam')
            model.fit(x=self.X_train.astype('float32'), y=self.y_train.astype('float32'),
                      epochs=100,
                      validation_split=0.25,
                      batch_size=16,
                      shuffle=True,
                      callbacks=[early_stopping])
            qr_upper = model

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
            qr_lower = GradientBoostingRegressor(alpha=alpha/2, loss='quantile', **self.hyperparameters['lower'])

            if self.verbose == 1:
                print(f"Fitting lower quantile ({alpha / 2})")
            qr_lower.fit(self.X_train, self.y_train)

            qr_upper = GradientBoostingRegressor(alpha=1-alpha/2, loss='quantile', **self.hyperparameters['upper'])
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
        if self.model is not None:
            p = self.model.predict(X.astype('float32'))
            pd_dict = {'y_true': y_test,
                       'y_median': p[:, 1],
                       'y_lower': p[:, 0],
                       'y_upper': p[:, 2]}
            test_pred_df = pd.DataFrame(pd_dict)
            test_pred_df.index = y_test.index
            return test_pred_df

        idx = X.index
        y_lower = self.qr_lower.predict(X.astype('float32'))
        y_upper = self.qr_upper.predict(X.astype('float32'))
        y_median = self.qr_med.predict(X.astype('float32'))

        if y_test is None:
            pd_dict = {'y_median': y_median,
                       'y_lower': y_lower,
                       'y_upper': y_upper}
        else:
            pd_dict = {'y_true': y_test,
                        'y_median': y_median.flatten(),
                       'y_lower': y_lower.flatten(),
                       'y_upper': y_upper.flatten()}
        # return pd_dict
        df = pd.DataFrame(pd_dict)
        df.index = idx
        return df

    # Calculate scores using X_cal and lower/upper quantiles. Return scores and
    # save to object as self.scores
    def calculate_scores(self):
        y_df = self.predict_quantiles(self.X_cal, y_test=self.y_cal)
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

def calc_coverage(cp_df):
    greq_lb = cp_df['y_true'] >= cp_df['y_lower']
    leq_ub = cp_df['y_true'] <= cp_df['y_upper']
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
