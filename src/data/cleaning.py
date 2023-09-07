"""
used to clean the data by reducing outliers/noise, handling missing values, etc.
"""

import os
import pandas as pd
# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to save the cleaned data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

techniques_df           = pd.readcsv (os.path.join (ROOT_FOLDER,'data/interim', 'collected_techniques_df.csv'))
techniques_mitigations_df          = pd.readcsv (os.path.join (ROOT_FOLDER,'data/interim', 'collected_mitigations_df.csv'))
groups_df               = pd.readcsv (os.path.join (ROOT_FOLDER,'data/interim', 'collected_groups_df.csv'))
groups_techniques_df    = pd.readcsv (os.path.join (ROOT_FOLDER,'data/interim', 'collected_groups_techniques_df.csv'))
groups_software_df      = pd.readcsv (os.path.join (ROOT_FOLDER,'data/interim', 'collected_groups_software_df.csv'))

DFS = {
    'techniques_df' : techniques_df,
    'techniques_mitigations_df' : techniques_mitigations_df,
    'groups_df': groups_df,
    'groups_techniques_df' : groups_techniques_df,
    'groups_software_df' : groups_software_df,
    }

FILTER_COLUMN_RENAME = {
    """
    each table is assigned with a tuple including:
        (1) a list of columns in the table that are used for training
        (2) a list of names for re-naming columns in (1) for clarity
    """
    'techniques_df' :           (['ID', 'platforms'],           ['technique_ID', 'platforms']), #only names for platforms, no IDs
    'techniques_mitigation_df': (['source ID', 'target ID'],    ['mitigation_ID', 'technique_ID']), 
    'groups_df' :               (['ID'],                        ['group_ID']),
    'groups_techniques_df':     (['source ID', 'target ID'],    ['group_ID', 'technique_ID']),
    'groups_software_df' :      (['source ID', 'target ID'],    ['group_ID', 'software_ID'])
}

def _batch_save_df_to_csv (file_name_dfs: dict, target_path, prefix =''):
    """
    Saves the DataFrames stored in a dict as csv file. 
    file_name_dfs: key = filenames, value = DataFrame
    
    """
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key
        if not filename.endswith (".csv"): filename+= ".csv"
        output_file = os.path.join(target_path, filename)
        
        df = file_name_dfs[key]
        df.to_csv (output_file, index = False)
    
    print ("Finished: files saved to", target_path)
    
    for key in file_name_dfs.keys():
        print ("\t", prefix + key, ".csv", sep = '')

def clean_data (target_path = TARGET_PATH):
    """
    Filters the selected columns for the collected data, then re-name them
    """    
    for key in DFS.keys():
        # 1- Filter the columns
        df = DFS[key]
        df = df[FILTER_COLUMN_RENAME[key][0]]
        # 2- Rename the columns
        df.columns = FILTER_COLUMN_RENAME[key][1]
        
        DFS[key] = df
    
    dfs = {
    "techniques_df" : techniques_df,
    "techniques_mitigations_df" : techniques_mitigations_df,
    "groups_df": groups_df,
    "groups_techniques_df" : groups_techniques_df,
    "groups_software_df" : groups_software_df,
    }
    _batch_save_df_to_csv (dfs, target_path, prefix =  'cleaned_')