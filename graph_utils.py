
import networkx as nx
import numpy as np
import scipy.sparse as sp
from datetime import timedelta
from math import sin, cos, sqrt, atan2, radians



def euclidian_dist(node_x, node_y):
    lat1 = node_x['lat']
    long1 = node_x['long']
    lat2 = node_y['lat']
    long2 = node_y['long']
    return np.sqrt((lat1-lat2)**2+(long1-long2)**2)

def distance_in_meters(node_x, node_y):
    R = 6373.0
    
    lat1 = radians(abs(node_x['lat']))
    lon1 = radians(abs(node_x['long']))
    lat2 = radians(abs(node_y['lat']))
    lon2 = radians(abs(node_y['long']))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance



def normalize_adj(adj, symmetric=True, add_self_loops=False):
    """
    Normalize adjacency matrix.
    Args:
        adj: adjacency matrix
        symmetric: True if symmetric normalization or False if left-only normalization
        add_self_loops: True if self loops are to be added before normalization, i.e., use A+I where A is the adjacency
            matrix and I is a square identity matrix of the same size as A.
    Returns:
        Return a sparse normalized adjacency matrix.
    """

    if add_self_loops:
        adj = adj + sp.diags(np.ones(adj.shape[0]) - adj.diagonal())

    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.float_power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def get_graph(data):
    G = nx.Graph()

    for station in data['ID'].unique():
        G.add_node(station)
        G.nodes[station]['ID'] = data[data['ID'] == station]['ID'].iloc[0]
        G.nodes[station]['lat'] = data[data['ID'] == station]['Latitude'].iloc[0]
        G.nodes[station]['long'] = data[data['ID'] == station]['Longitude'].iloc[0]
        G.nodes[station]['pos'] = (G.nodes[station]['long'], G.nodes[station]['lat'])


    for node_x in G.nodes:
        for node_y in G.nodes:
            dist = distance_in_meters(G.nodes[node_x], G.nodes[node_y])
            if (dist > 2.5):
                continue
            G.add_edge(node_x, node_y)
            G[node_x][node_y]['weight'] = np.exp(-dist)


    adj = nx.adjacency_matrix(G)
    return G, normalize_adj(adj).todense()


def setup_GCN(data, forecast_horizon, num_lags, nx = 5, ny = 5):

    G, adj = get_graph(data)

    number_of_hours = int((data.index.max()-data.index.min()).total_seconds()//(3600*24))

    timeseries_ = np.zeros([len(G.nodes()), number_of_hours+1])
    start_time = data.index.min()

    for i in range(0, number_of_hours+1):
        timewindow_start = start_time+timedelta(days=i)
        
        current = data[(data.index == timewindow_start)]

        for k, node in enumerate(G.nodes()):
            tmp = current[G.nodes[node]['ID']==current['ID']]
            timeseries_[k,i] = np.sum(tmp['Energy'])


    timeseries_[timeseries_ == 0] = np.random.normal(np.zeros_like(timeseries_[timeseries_ == 0]),5)
    max_ =np.max(timeseries_, axis=1)
    normalized = timeseries_/ np.max(timeseries_, axis=1)[:,None]


    NUM_LAGS = num_lags
    STEPS_AHEAD = forecast_horizon
    n_nodes = len(G.nodes())

    matrix_lags = np.zeros((timeseries_.shape[-1]-(NUM_LAGS+STEPS_AHEAD), timeseries_.shape[0], NUM_LAGS+STEPS_AHEAD)) 
    i_train = matrix_lags.shape[0]-STEPS_AHEAD
    i_test = matrix_lags.shape[0]

    for i in range(matrix_lags.shape[0]):
        matrix_lags[i] = normalized[:,i:i+NUM_LAGS+STEPS_AHEAD]


    # ---------------- Train/test split
    X_train = np.zeros((i_train,n_nodes,NUM_LAGS))
    y_train = np.zeros((i_train,n_nodes,STEPS_AHEAD))
    X_test = np.zeros((i_test-i_train,n_nodes,NUM_LAGS))
    y_test = np.zeros((i_test-i_train,n_nodes,STEPS_AHEAD))

    for i, node in enumerate(G.nodes):
        X_train[:,i,:] = matrix_lags[:i_train,i, :NUM_LAGS]
        y_train[:,i,:] = matrix_lags[:i_train,i, NUM_LAGS:]
        X_test[:,i,:] = matrix_lags[i_train:i_test,i,:NUM_LAGS]
        y_test[:,i,:] = matrix_lags[i_train:i_test,i,NUM_LAGS:]


    return G, adj, X_train, y_train, X_test, y_test