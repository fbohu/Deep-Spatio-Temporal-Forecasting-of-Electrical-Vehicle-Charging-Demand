from data_utils import *

import numpy as np

def process_data(data, model_name, prediction_horizont, num_lags, nx=5,ny=5):

    raster_map = get_raster_map(data)

    max_ =np.max(raster_map, axis=0)+0.01
    normalized = raster_map/max_

    normalized[np.isnan(normalized)] = 0
    normalized[normalized == 0] = np.random.normal(np.zeros_like(normalized[normalized == 0]),0.01)

    matrix_lags = np.zeros((normalized.shape[0]-(num_lags+prediction_horizont), num_lags+prediction_horizont, nx, ny))
    
    i_train = matrix_lags.shape[0]-prediction_horizont
    i_test = matrix_lags.shape[0]

    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[i:i+num_lags+prediction_horizont,:,:]

    # ---------------- Train/test split
    X_train = np.zeros((i_train,num_lags,nx,ny))
    y_train = np.zeros((i_train,prediction_horizont, nx,ny))
    X_test = np.zeros((i_test-i_train,num_lags,nx,ny))
    y_test = np.zeros((i_test-i_train,prediction_horizont, nx,ny))

    for i in range(nx):
        for j in range(ny):
            X_train[:,:,i,j] = matrix_lags[:i_train,:num_lags,i,j]
            y_train[:,:,i,j] = matrix_lags[:i_train,num_lags:,i,j]
            X_test[:,:,i,j] = matrix_lags[i_train:i_test, :num_lags,i,j]
            y_test[:,:,i,j] = matrix_lags[i_train:i_test,num_lags:,i,j]


    if model_name =='convlstm':
        lstm_X_train = X_train[:,:,:,:,np.newaxis]
        lstm_X_test = X_test[:,:,:,:,np.newaxis]

        return lstm_X_train, y_train, lstm_X_test, y_test

    X_train = np.moveaxis(X_train,1,-1)
    y_train = np.moveaxis(y_train,1,-1)
    X_test = np.moveaxis(X_test,1,-1)
    y_test = np.moveaxis(y_test,1,-1)


    return X_train, y_train, X_test, y_test