import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.layers import InputLayer
from keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.callbacks import (EarlyStopping, History,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)


def jm_cnn_model(X_train, targets, nfilters, kernel_length,
                 loss="binary_crossentropy"):
    """The model that Jean Michel has implemented, and trained.
    """
    dropout = 0.2
    dropout = 0.01
    K.set_image_data_format('channels_last')

    multi_layer_keras_model = tf.keras.Sequential()
    multi_layer_keras_model.add(Input(X_train.shape[1:]))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(1, activation='sigmoid'))
    # multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
    #                                loss='mean_squared_logarithmic_error')
    multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
                                    loss=loss,
                                    metrics=[metrics.MSE])
    multi_layer_keras_model.summary()
    return multi_layer_keras_model

def jm_cnn_model_beta(X_train, targets, nfilters, kernel_length,
                 loss="mse"):
    """Some slight modifications on the model that Jean Michel has implemented,
    and trained. this nn can be a costumized to compare the effect of optimizers,
    dropout, and activation functions. 
    """
    dropout = 0.2
    dropout = 0.01
    K.set_image_data_format('channels_last')
    multi_layer_keras_model = tf.keras.Sequential()
    multi_layer_keras_model.add(Input(X_train.shape[1:]))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(1, activation='linear'))
    multi_layer_keras_model.compile(optimizer='adam',
                                    loss=loss,
                                    metrics=[metrics.MSE])
    multi_layer_keras_model.summary()
    return multi_layer_keras_model

