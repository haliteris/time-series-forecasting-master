import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import keras
from math import sqrt


def check_gpu():  
    #Involving the GPU.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    #tf.config.set_visible_devices([], 'CPU') # hide the CPU
    #tf.config.set_visible_devices(gpus[0], 'GPU') # unhide potentially hidden GPU
    #tf.config.get_visible_devices()

def set_LSTM(neuron,n_dense,n_epoch, b_size, train_X, test_X, train_y, test_y):
    # Design network, Long Short-term Memory (LSTM)
    model = Sequential()
    model.add(LSTM(neuron, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(n_dense))
    model.compile(loss='mae', optimizer='adam')
    # Fit network
    history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=b_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # Plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def save_model(path):
    #Save model to chosen path.
    model.save(path)
    
def load_model(path):
    #Load the saved-model at the path.
    model = keras.models.load_model(path)
    print('Model has been loaded.')
    return model
    
def evaluation(test_X, test_y, cmodel):
    #Prediction over test set from saved-model.
    predicted = cmodel.predict(test_X)
    #Calculating Mean-saured-error.
    rmse = sqrt(mean_squared_error(test_y, predicted))
    print('Test RMSE: %.3f' % rmse)
    result = r2_score(test_y, predicted)
    print('Total variance explained by model / Total variance : ',result)
    