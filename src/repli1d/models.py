from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import (EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau)


def jm_cnn_model(X_train, targets, nfilters, kernel_length,
                 loss="binary_crossentropy"):
    """The model that Jean Michel has been implemented, and trained.
    """
    dropout = 0.2
    dropout = 0.01
    K.set_image_data_format('channels_last')

    multi_layer_keras_model = Sequential()
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(len(targets[0])))
    multi_layer_keras_model.add(Activation("sigmoid"))
    # multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
    #                                loss='mean_squared_logarithmic_error')
    multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
                                    loss=loss)
    multi_layer_keras_model.summary()
    return multi_layer_keras_model

def cnn_model(X_train, targets, nfilters, kernel_length,
                 loss="mse"):
    """The model that Jean Michel has been implemented, and trained.
    """
    dropout = 0.2
    dropout = 0.01
    K.set_image_data_format('channels_last')

    multi_layer_keras_model = Sequential()
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), input_shape=X_train.shape[1:]))
    multi_layer_keras_model.add(Activation('relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(len(targets[0])))
    multi_layer_keras_model.add(Activation("linear"))
    # multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
    #                                loss='mean_squared_logarithmic_error')
    multi_layer_keras_model.compile(optimizer='adam',
                                    loss=loss)
    multi_layer_keras_model.summary()
    return multi_layer_keras_model


