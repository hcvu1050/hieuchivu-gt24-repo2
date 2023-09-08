"""
used to clean the data by reducing outliers/noise, handling missing values, etc.
1. Reads collected files from data/interim
2. Filters the comlumns needed for training
3. Rename the columns
4. Main Function clean_data: Merge the tables into 3 main tables
    (a). Group-Technique interaction matrix
    (b). Technique features
    (c). Group features
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

# Get the collected tables
techniques_df               = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_techniques_df.csv'))
techniques_mitigations_df   = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_techniques_mitigations_df.csv'))
groups_df                   = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_groups_df.csv'))
groups_techniques_df        = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_groups_techniques_df.csv'))
groups_software_df          = pd.read_csv (os.path.join (SOURCE_PATH, 'collected_groups_software_df.csv'))


"""
FILTER_COLUMN_RENAME: used to filter the columns needed for training from the collected table
key = filename for a table, value = tuple
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
    'im_positive_cases_df':         (groups_techniques_df,      ['source ID', 'target ID'],     ['group_ID', 'technique_ID'])
    #im = interation matrix
}

### END OF CONFIGURATION ###


def _filter_rename_columns ():
    """
    Based on FILTER_COLUMN_RENAME:\n
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

def _make_interaction_matrix (user_IDs_df = groups_df, 
                              item_IDs_df = techniques_df, 
                              positive_cases = groups_techniques_df) -> pd.DataFrame():
    """Creates an interaction matrix (all possible combination) between users and items based on the IDs.

    """
    group_technique_interactions = pd.merge (user_IDs_df, item_IDs_df, how = 'cross')
    positive_cases ['target'] = 1
    group_technique_interaction_matrix = pd.merge (
        left = group_technique_interactions,
        right = positive_cases, 
        on = ['group_ID', 'technique_ID'], 
        how = 'left'
    )
    group_technique_interaction_matrix['target'].fillna (0, inplace= True)
    return group_technique_interaction_matrix

def _combine_features (object: str, dfs: dict) -> pd.DataFrame():
    """Combines the features of the object (Group or Technique) based of the tables of features stored in dfs. 
    The features are merged based on the list of object IDs (group_ID or technique_ID).

    Args:
        object (str): "group" or "technique"
        dfs (dict): dfs stores the filtered tables for the object

    Returns:
        pd.DataFrame: Return the merged table of the object
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
    interaction_matrix = _make_interaction_matrix(
        user_IDs_df= filtered_dfs['groups_df'],
        item_IDs_df= filtered_dfs['techniques_df'],
        positive_cases= filtered_dfs['im_positive_cases_df']
    )
    
    res_dfs = {
        'technique_features_df' : technique_features_df,
        'group_features_df' : group_features_df ,
        'interaction_matrix' : interaction_matrix,
    }
    utils.batch_save_df_to_csv (res_dfs, target_path,prefix= 'cleaned_')
    return res_dfs