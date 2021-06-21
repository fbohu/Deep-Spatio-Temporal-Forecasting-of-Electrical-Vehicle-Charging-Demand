import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def find_gridcell(row, x_list, y_list):
        for i, x_cell in enumerate(x_list):
            if row['Longitude'] <= x_cell:
                x_grid_cell = i-1
                break

        for i, y_cell in enumerate(y_list):
            if row['Latitude'] <= y_cell:
                y_grid_cell = i-1
                break

        return x_grid_cell, y_grid_cell


def get_grid_cells(data, nx=5, ny=5):
    x_ = np.linspace(min(data['Longitude'])-0.00005, max(data['Longitude'])+0.00005,nx+1)
    y_ = np.linspace(min(data['Latitude'])-0.00005, max(data['Latitude'])+0.00005,ny+1)
    print(x_)
    print(y_)
    data['x_cell'] = -1
    data['y_cell'] = -1
    for index, row in data.iterrows():
        data.at[index,'x_cell'], data.at[index,'y_cell'] = find_gridcell(row, x_, y_)

    return data

def get_raster_map(data, normalized=False, verbose = False, nx=5, ny =5):

    number_of_hours = int((data.index.max()-data.index.min()).total_seconds()//(3600*24))
    raster_map = np.zeros([number_of_hours+1, nx, ny])
    start_time = data.index.min()

    for i in range(0, number_of_hours+1):
        timewindow_start = start_time+timedelta(days=i)
        
        current = data[ (data.index == timewindow_start)]


        for x in range(0, nx):
            for y in range(0, ny):
                no_chargers = len(current[(current.x_cell == x) & (current.y_cell == y)]['ID'].unique())
                if no_chargers == 0:
                    continue
                raster_map[i,x,y] = np.sum(current[(current.x_cell == x) & (current.y_cell == y)]['Energy'])

        if (verbose) and ((i % 50)==0):
            print("Done with {} out of {}".format(i, number_of_hours))

    
    if normalized:
        max_ =np.max(raster_map, axis=0)+0.01
        raster_map = raster_map/max_
    
    return raster_map
