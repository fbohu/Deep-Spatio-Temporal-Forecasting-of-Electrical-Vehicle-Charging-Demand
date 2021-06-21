from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import VAR

import numpy as np

def get_baselines(forecast_horizon, normalized_raster_map, data):

    n_size = normalized_raster_map.shape[0]
    var_raster_map = np.reshape(normalized_raster_map,(n_size,-1))
    test = var_raster_map[-(forecast_horizon+1):-1]
    train = var_raster_map[:(n_size-(forecast_horizon+1))]
    xs, ys = np.where(train == -1)
    train[xs,ys] = np.random.normal(np.zeros(len(xs))-1,0.001)
    train[xs,ys]
    model = VAR(train)
    model_fit = model.fit(maxlags=21)
    forecast_ = model_fit.forecast(train, forecast_horizon).reshape(-1,5,5)
    train = train.reshape(-1,5,5)
    test = test.reshape(-1,5,5)



    ts = data['Energy'].groupby(data.index).sum()
    ts_test = ts[-(forecast_horizon+1):-1].values
    ts_train = ts[:-(forecast_horizon+1)].values

    mod = AutoReg(ts_train, 21, old_names=False)
    fit_ = mod.fit()
    preds = fit_.predict(ts_train.shape[0], ts_train.shape[0]+(forecast_horizon-1))

    # Return the VAR forecast and the AR
    return forecast_, preds