import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, GaussianNoise, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow import keras
from keras.callbacks import *
import tensorflow as tf

class NN_Estimator():
    def __init__(self, n_units, n_layers, alpha):
        self.n_units = n_units
        self.n_layers = n_layers
        self.alpha = alpha
        self.model = None

    def fit(self, X, y):
        input_dim = X.shape[1]
        n_units = [self.n_units] * self.n_layers
        activations = ['relu'] * self.n_layers
        gauss_std = [0] * self.n_layers

        # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        self.model = create_nn_model(input_dim, n_units, activations, gauss_std=gauss_std)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        self.model.compile(loss=lambda y_t, y_p: qloss_nn(y_true=y_t, y_pred=y_p, q=self.alpha),
                      optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3))
        self.model.fit(x=X.astype('float32'), y=y.astype('float32'),
                  epochs=50,
                  validation_split=0.25,
                  batch_size=16,
                  shuffle=True,
                  callbacks=[early_stopping])
        return self

    def set_params(self, n_units, n_layers, alpha):
        self.n_units = n_units
        self.n_layers = n_layers
        self.alpha = alpha
        return self

    def get_params(self, deep=True):
        return {'n_units': self.n_units, 'n_layers': self.n_layers, 'alpha': self.alpha}

    def predict(self, X):
        preds = self.model.predict(X.astype('float32'))
        return preds

    # def score(self, X):
    #     preds = self.predict(X)
    #     return d2_pinball_score(y_test, y_hat)

@tf.autograph.experimental.do_not_convert
def qloss_nn(y_true, y_pred, q=0.5):
    left = q * (y_true - y_pred)
    right = (q-1) * (y_true - y_pred)
    return K.maximum(left, right)

def create_nn_model(input_dim, num_units, activations, gauss_std=None):
    input = Input((input_dim,), name='input')
    n_layers = len(num_units)

    if gauss_std is None:
        gauss_std = np.zeros(n_layers)

    x = input
    for i in range(n_layers):
        # x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                  # kernel_regularizer=regularizers.l2(0.001), activation=act[i])(x)

        x = Dense(num_units[i], use_bias=True, kernel_initializer='he_normal', bias_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(0.001), activation=activations[i])(x)
        # x = Dropout(dp[i])(x)
        x = GaussianNoise(gauss_std[i])(x)

    x = Dense(1, activation=None, use_bias=True, kernel_initializer='he_normal',
              bias_initializer='he_normal')(x)

    model = Model(input, x)
    return model

