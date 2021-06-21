from models.gcnlstm import get_gcnlstm
from models.convlstm import get_conv_lstm
from models.conv import get_conv_model



def get_model(name, args):
    if name == 'conv':
        return get_conv_model(args)
    elif name == 'convlstm':
        return get_conv_lstm(args)



def get_model_args(name, pred_horizont,X_train=None, adj_matrix=None):
    if name == 'conv':
        return get_conv_args(pred_horizont, X_train)
    elif name == 'convlstm':
        return get_conv_lstm_args(pred_horizont, X_train)



def get_conv_args(pred_horizont, X_train):
    return {"filters": 16, "forecast_horizon": pred_horizont, "input_shape": X_train[0].shape,
            "lr": 1e-4, "epochs":500}

def get_conv_lstm_args(pred_horizont, X_train):
    return {"filters": 16, "forecast_horizon": pred_horizont, "input_shape": X_train[0].shape,
            "lr": 1e-4, "epochs":500, "lstm_size": 100}