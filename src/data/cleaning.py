"""
used to clean the data by reducing outliers/noise, handling missing values, etc.
1. Read collected files from data/interim
2. Filter the important columns
3. Rename the columns
4. Merge the tables into 3 main tables
    a. Technique features
    b. Group features
    c. Target Group-Technique
5. Export to data/interim
"""

import os
import pandas as pd
from . import utils
### CONFIGURATION ###

# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get collected data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the cleaned data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

techniques_df               = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_techniques_df.csv'))
techniques_mitigations_df   = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_techniques_mitigations_df.csv'))
groups_df                   = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_groups_df.csv'))
groups_techniques_df        = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_groups_techniques_df.csv'))
groups_software_df          = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_groups_software_df.csv'))


"""
FILTER_COLUMN_RENAME: key = filename for a table, value = tuple
each table is assigned with a tuple including:
    (1) the dataframe 
    (2) a list of columns in the table that are used for training
    (3) a list of names for re-naming columns in (1) for clarity
"""
FILTER_COLUMN_RENAME = {
    'techniques_df' :               (techniques_df,             ['ID'],                         ['technique_ID']), 
    'techniques_platforms_df' :     (techniques_df,             ['ID', 'platforms'],            ['technique_ID', 'platforms']), #only names for platforms, no IDs
    'techniques_mitigations_df':    (techniques_mitigations_df, ['source ID', 'target ID'],     ['mitigation_ID', 'technique_ID']), 
    'groups_df' :                   (groups_df,                 ['ID'],                         ['group_ID']),
    'groups_software_df' :          (groups_software_df,        ['source ID', 'target ID'],     ['group_ID', 'software_ID']),
    'target_groups_techniques_df':  (groups_techniques_df,      ['source ID', 'target ID'],     ['group_ID', 'technique_ID'])
}

### END OF CONFIGURATION ###


def _filter_rename_columns ():
    """
    Filters the selected columns for the collected data, then re-name them
    """
    res_dfs = {}
    for key in FILTER_COLUMN_RENAME.keys():
        # 1- Filter the columns
        df = FILTER_COLUMN_RENAME[key][0]        
        df = df[FILTER_COLUMN_RENAME[key][1]]
        # 2- Rename the columns
        df.columns = FILTER_COLUMN_RENAME[key][2]
        
        res_dfs[key] = df
    return res_dfs

def _combine_features (object: str, dfs: dict):
    """Combines features of Group or Technique based of the tables of features stored in dfs. 
    The features are merged with the list of object IDs (group_ID or technique_ID)

    Args:
        object (str): "group" or "technique"
        dfs (dict): dfs stores the filtered columns in collected tables
    """
    object_features = pd.DataFrame()
    id_name = ''
    if object == 'group':
        group_IDs = dfs['groups_df']
        object_features = group_IDs 
        id_name = 'group_ID'
    elif object == 'technique':
        technique_IDs = dfs['techniques_df']
        object_features = technique_IDs
        id_name = 'technique_ID'
    
    # The features are merged with the list of object IDs (group_ID or technique_ID)
    for key in [key for key in dfs.keys() if key.startswith (object)]:
        object_features = pd.merge (
            left = object_features,
            right= dfs[key],
            on = id_name,
            how = 'left'
        )
    return object_features

def clean_data(target_path = TARGET_PATH):
    """Filters the columns needed for training, then combines all features of a object group into one table.\n
    Returns 3 tables:\n
    a. Technique features\n
    b. Group features\n
    c. Target Group-Technique\n
    """
    filtered_dfs = _filter_rename_columns()
    group_features_df = _combine_features (object= 'group', dfs = filtered_dfs)
    technique_features_df = _combine_features (object= 'technique', dfs = filtered_dfs)
    target_groups_techniques_df = filtered_dfs['target_groups_techniques_df']
    
    res_dfs = {
        'technique_features_df' : technique_features_df,
        'group_features_df' : group_features_df ,
        'target_groups_techniques' : target_groups_techniques_df
    }
    utils.batch_save_df_to_csv (res_dfs, target_path,prefix= 'cleaned_')