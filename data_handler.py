import pandas as pd
from data_utils import *


def get_dataset(name):
    if name == 'Palo Alto':
        return get_palo_alto()
    elif name == 'Boulder':
        return get_boulder()
    elif name == 'Dundee':
        return get_dundee()
    elif name == 'Perth':
        return get_perth()


def get_palo_alto():
    Palo_alto = pd.read_csv("data/Palo Alto/ChargePoint Data CY20Q4.csv")

    Palo_alto = get_grid_cells(Palo_alto)

    Palo_alto['End'] = pd.to_datetime(Palo_alto['Start Date'], format='%m/%d/%Y %H:%M')+pd.to_timedelta(Palo_alto['Charging Time (hh:mm:ss)'])
    Palo_alto['Start'] = pd.to_datetime(Palo_alto['Start Date'], format='%m/%d/%Y %H:%M')
    Palo_alto['Start'] = Palo_alto['Start'].dt.floor('D')
    Palo_alto = Palo_alto.set_index('Start').sort_index()

    Palo_alto['Energy'] = Palo_alto['Energy (kWh)']
    Palo_alto['ID'] = Palo_alto['Station Name']
    return Palo_alto

def get_boulder():
    Boulder = pd.read_csv("data/Boulder/Electric_Vehicle_Charging_Station_Energy_Consumption.csv")
    
    Boulder = Boulder[Boulder['Energy__kWh_'] > 0]
    Boulder = Boulder[Boulder['Energy__kWh_'] < 2000]

    Boulder_locations = pd.read_csv("data/Boulder/Boulder_locations.csv")
    Boulder_locations = get_grid_cells(Boulder_locations)

    Boulder = pd.merge(Boulder, Boulder_locations, on='Address', how = 'left')

    Boulder['Start'] = pd.to_datetime(Boulder['Start_Date___Time'], format='%Y/%m/%d %H:%M:%S').dt.round('D')
    Boulder['End'] = pd.to_datetime(Boulder['End_Date___Time'], format='%Y/%m/%d %H:%M:%S')

    Boulder = Boulder.set_index('Start').sort_index()
    Boulder['Energy'] = Boulder['Energy__kWh_']
    Boulder['ID'] = Boulder['Station_Name']
    return Boulder

def get_dundee():
    Dundee_1 = pd.read_csv("data/Dundee/charge-sessions-june-sept.csv")
    Dundee_2 = pd.read_csv("data/Dundee/cp-data-dec-mar-2018.csv")
    Dundee_3 = pd.read_csv("data/Dundee/cp-data-mar-may-2018.csv")
    Dundee_4 = pd.read_csv("data/Dundee/cpdata.csv")

    Dundee_locations = pd.read_csv("data/Dundee/cp-locations_enriched.csv")
    Dundee_locations = get_grid_cells(Dundee_locations)
    Dundee = Dundee_1.append([Dundee_2, Dundee_3, Dundee_4])
    Dundee = Dundee.reset_index()

    Dundee = Dundee[Dundee['Site'].notna()]
    Dundee = Dundee[Dundee['Total kWh'].notna()]
    Dundee = Dundee[Dundee['Total kWh'] < 1000]
    Dundee = Dundee[Dundee['Total kWh'] > 0]
    Dundee['CP ID'] = Dundee['CP ID'].astype(int).astype(str)
    Dundee = pd.merge(Dundee, Dundee_locations, on='CP ID', how = 'left')
    Dundee = Dundee[~Dundee['x_cell'].isna()]

    Dundee = Dundee[Dundee['Site']!='***TEST SITE*** Charge Your Car HQ']
    Dundee['Start'] = pd.to_datetime(Dundee['Start Date']+" " + Dundee['Start Time'], format='%d/%m/%Y %H:%M')
    Dundee['End'] = pd.to_datetime(Dundee['End Date']+" " + Dundee['End Time'], format='%d/%m/%Y %H:%M')

    Dundee['Duration'] = Dundee['End'] - Dundee['Start']
    Dundee['Start'] = Dundee['Start'].dt.floor('D')

    Dundee = Dundee.set_index('Start').sort_index()
    Dundee = Dundee[~Dundee['Total kWh'].isna()]

    Dundee = Dundee[~Dundee['Total kWh'].isna()]
    Dundee = Dundee[Dundee['Duration'].dt.total_seconds() > 0]
    #Dundee = Dundee[Dundee['Duration'].dt.total_seconds() < 60*60*36]


    Dundee['Energy'] = Dundee['Total kWh']
    Dundee['ID'] = Dundee['Site']
    return Dundee


def get_perth():

    Perth_1 = pd.read_csv('../Datasets/Perth/sept17toaug18standardisedcorrected.csv')
    Perth_2 = pd.read_csv('../Datasets/Perth/sept18toaug19standardisedcorrected.csv')
    Perth_3 = pd.read_csv('../Datasets/Perth/electricvehiclechargecorrected.csv')
    Perth_location = pd.read_csv("../Datasets/Perth/perth_locations.csv")

    Perth_location = get_grid_cells(Perth_location)

    Perth = Perth_1.append([Perth_2, Perth_3])
    Perth = Perth.reset_index()

    Perth['Start'] = pd.to_datetime(Perth['Start Date']+" " + Perth['Start Time'], format='%d/%m/%Y %H:%M')
    Perth['End'] = pd.to_datetime(Perth['End Date']+" " + Perth['End Time'], format='%d/%m/%Y %H:%M')
    Perth['Start'] = Perth['Start'].dt.floor('D')
    Perth['Site'] = Perth['Site'].str.replace(", ","-")
    Perth = Perth[Perth['Total kWh'] > 0]


    Perth = Perth[Perth['Site']!='***TEST SITE*** Charge Your Car HQ']

    Perth = pd.merge(Perth, Perth_location, on='Site', how = 'left')
    Perth = Perth.set_index('Start').sort_index()

    Perth['Energy'] = Perth['Total kWh']
    Perth['ID'] = Perth['Site']

    return Perth
