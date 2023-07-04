import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector as sqlcn

resampling_rules = {
    'ext1' : {'air_temp' : np.mean,
                 'apperture' : np.max,
                #'aprtr_change': np.sum
                },
    'ext2' : {'air_temp' : np.mean,
                 'apperture' : np.mean,
                #'aprtr_change': np.sum
                },
    'rta' : { 't1': np.mean,
                't2': np.mean,
                't3': np.mean,
                't4': np.mean,
                't5': np.mean,
                't6': np.mean,
                't7': np.mean,
                't8': np.mean,
                't9': np.mean,
                't11': np.mean,
                't11': np.mean},
    'ib_facade' : { 'air_temp' : np.mean,
                      'r_humid': np.mean },
    'ib_roof' : { 'air_temp': np.mean,
                    'r_humid': np.mean },
    'libelium' : { 'epoch_time': np.mean,
                  'battery_level': np.mean,
                  'dendrometer': np.mean,
                  'air_temp': np.mean,
                  'r_humid': np.mean,
                  'atm_p': np.mean,
                  'rock_temp': np.mean,
                  'voltage': np.mean},
    'arduino': {'t1': np.mean,
                't2': np.mean,
                't3': np.mean,
                't4': np.mean,
                't5': np.mean,
                't6': np.mean,
                'luminosity': np.mean,
                't1_3dht11': np.mean,
                't2_3dht11': np.mean,
                't3_3dht11': np.mean,
                'h1_3dht11': np.mean,
                'h2_3dht11': np.mean,
                'h3_3dht11': np.mean}
}

def download_data():
    conn = sqlcn.connect(host='geosens.rocks',
                     database='wzvsrfmy_egypt_sensors',
                     user= "wzvsrfmy_TT95rt",
                     password= "TT95S@Q")
    
    raw_data = {
    'ext1': pd.read_sql('SELECT * FROM extensometer_1', con=conn),
    'ext2': pd.read_sql('SELECT * FROM extensometer_2', con=conn),
    'rta': pd.read_sql('SELECT * FROM rock_temp_array', con=conn),
    'ib_facade': pd.read_sql('SELECT * FROM ib_facade', con=conn),
    'ib_roof': pd.read_sql('SELECT * FROM ib_roof', con=conn),
    'libelium': pd.read_sql('SELECT * FROM libelium', con=conn),
    'arduino': pd.read_sql('SELECT * FROM arduino_tt95', con=conn)
    }    
    
    for key in raw_data.keys():
        raw_data[key] = raw_data[key].drop_duplicates(ignore_index=True)
        raw_data[key].set_index('datetime', inplace=True)
    
    return raw_data

# Function that returns data for each sensor within specified dates
def select_date_range(start_date=None, stop_date=None, df_dict=None):
    return_dict = {}
    for key in df_dict.keys():
        return_df = df_dict[key].sort_index().loc[start_date:stop_date, :]
        if not return_df.empty:
            return_dict[key] = return_df
    return return_dict

# Function that resamples each df in the dictionary based on the 
# specified interval; the resampling blueprint is a nested dictionary
# containing the resampling functions for each column of each dataframe
def resample_dfs(raw_dfs, interval, rsmpl_blueprint):
    resampled = {}
    for k in raw_dfs.keys():
        resampled[k] = raw_dfs[k].resample(interval).agg(rsmpl_blueprint[k], axis=1)
    return resampled

# Function to join data of all sensors (input: dictionary containing sensor dataframes)
# based on common datetimes. It is recommended to first select a date range that 
# all sensors have in common and resample it at an appropriate interval before
# joining all data to avoid missing values
def join_data(df_dict, df_keys=None, exclude=None):
    renamed = {}
    keys = []
    if df_keys:
        keys = df_keys
    else:
        keys = list(df_dict.keys())
    print(keys)
    if exclude:
        keys.remove(exclude)
    
    for k in keys:
        old_col_names = df_dict[k].columns.values
        col_maping = {}
        for old_name in old_col_names:
            col_maping[old_name] = f'{old_name}_{k}'
        renamed[k] = df_dict[k].rename(col_maping, axis=1)

    return pd.concat([renamed[k] for k in keys], axis=1, join='inner').dropna()

# Helper function to quickly plot an indexed (datetime) dataframe
def plot_variables(df, variable_list, xlabel=None, ylabel=None):  
    df.loc[:, variable_list].plot()

def join_data_v2(df_dict):
    df_combined = pd.concat(df_dict.values(), keys=df_dict.keys(), axis=1)
    df_combined.columns.droplevel(1)
    return df_combined