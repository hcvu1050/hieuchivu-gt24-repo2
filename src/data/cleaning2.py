"""
V2.0
used to clean the data by reducing outliers/noise, handling missing values, etc.
1. Reads collected files from data/interim
2. Filters the comlumns needed for training
3. Rename the columns
4. Main Function clean_data: Merge the tables into 3 main tables
    (a). Group-Technique interaction matrix
    (b). Technique features: Containing all possible features for Techniques. 
    (c). Group features: Containing all possible features for Groups
5. Export to data/interim
"""
import os
import pandas as pd
from . import utils
from .constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME
# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get collected data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the cleaned data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)

### END OF CONFIGURATION ###

def _get_data():
    """Get the collected tables and make settings for filtering and renaming columns
    
    Returns a dictionary: `data_and_setting` used to filter the columns needed for training from the collected tables
    key = filename for a table, value = tuple
    each table is assigned with a tuple including:
        (1) the dataframe 
        (2) a list of columns in the table that are used for training
        (3) a list of names for re-naming columns in (1) for clarity
    """
    groups_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_groups_df.csv'))
    groups_software_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_groups_software_df.csv'))
    techniques_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_df.csv'))
    techniques_mitigations_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_mitigations_df.csv'))
    techniques_detections_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_detections_df.csv'))
    techniques_software_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_techniques_software_df.csv'))
    
    labels_df = pd.read_csv(os.path.join (SOURCE_PATH, 'collected_labels_df.csv'))
    
    data_and_setting = {
        'groups_df' :                   (groups_df,         ['ID'],                     [GROUP_ID_NAME]),
        'groups_software_df' :          (groups_software_df,['source ID', 'target ID'], [GROUP_ID_NAME, 'software_ID']),
        'techniques_df' :               (techniques_df,     ['ID'],                     [TECHNIQUE_ID_NAME]), 
        'techniques_platforms_df':      (techniques_df,     ['ID', 'platforms'],        [TECHNIQUE_ID_NAME, 'platforms']),
        'techniques_tactics_df':        (techniques_df,     ['ID', 'tactics'],          [TECHNIQUE_ID_NAME, 'tactics']),
        'techniques_defenses_bypassed_df':      (techniques_df, ['ID', 'defenses bypassed'],    [TECHNIQUE_ID_NAME, 'defenses_bypassed']),
        'techniques_permissions_required_df':   (techniques_df, ['ID','permissions required'],  [TECHNIQUE_ID_NAME, 'permissions required']),
        'techniques_mitigations_df':    (techniques_mitigations_df, ['source ID', 'target ID'],     ['mitigation_ID', TECHNIQUE_ID_NAME]), 
        'techniques_detections_df' :    (techniques_detections_df,  ['source name','target ID'],    ['detection name', TECHNIQUE_ID_NAME]),
        'techniques_software_df':       (techniques_software_df,    ['source ID', 'target ID'],     ['software_ID', TECHNIQUE_ID_NAME]),
        'labels_df' :                   (labels_df,                 ['source ID', 'target ID'],     [GROUP_ID_NAME, TECHNIQUE_ID_NAME])
    }
    return data_and_setting

def _filter_rename_columns (data_and_setting):
    """
    Based on data_and_setting:\n
    Filters the selected columns for the collected data, then re-name them
    """
    res_dfs = {}
    for key in data_and_setting.keys():
        df = data_and_setting[key][0]        
        # 1- Filter the columns
        df = df[data_and_setting[key][1]]
        # 2- Rename the columns
        df.columns = data_and_setting[key][2]
        
        res_dfs[key] = df
    return res_dfs

def _make_interaction_matrix (user_IDs_df, 
                              item_IDs_df, 
                              positive_cases) -> pd.DataFrame():
    """Creates an interaction matrix (all possible combination) between users and items based on the IDs.

    """
    group_technique_interactions = pd.merge (user_IDs_df, item_IDs_df, how = 'cross')
    # positive_cases ['target'] = 1
    positive_cases = positive_cases.assign (label = 1)
    group_technique_interaction_matrix = pd.merge (
        left = group_technique_interactions,
        right = positive_cases, 
        on = ['group_ID', 'technique_ID'], 
        how = 'left'
    )
    group_technique_interaction_matrix[LABEL_NAME].fillna (0, inplace= True)
    return group_technique_interaction_matrix

def _combine_features (object: str, dfs: dict) -> pd.DataFrame():
    """Combines the features of the object (Group or Technique) based of the tables of features stored in dfs. 
    In dfs, the key indicates if its value belongs to Group or Technique based on the key's prefix.
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
        object_features = group_IDs #initialize the result dataframe. starts with the list of IDs
        id_name = GROUP_ID_NAME
    elif object == 'technique':
        technique_IDs = dfs['techniques_df'] #initialize the result dataframe. starts with the list of IDs
        object_features = technique_IDs
        id_name = TECHNIQUE_ID_NAME
    
    # The features are merged with the list of object IDs (group_ID or technique_ID)
    for key in [key for key in dfs.keys() if key.startswith (object)]:
        object_features = pd.merge (
            left = object_features,
            right= dfs[key],
            on = id_name,
            how = 'left'
        )
    return object_features

def clean_data(target_path = TARGET_PATH, save_as_csv = True):
    """Filters the columns needed for training, then combines all features of a object group into one table.\n
    Returns 3 tables:\n
    a. Technique features\n
    b. Group features\n
    c. Target Group-Technique\n
    """
    print (PROCESS_RUNNING_MSG)
    data_and_setting = _get_data()
    filtered_dfs = _filter_rename_columns(data_and_setting)
    
    group_features_df = _combine_features (object= 'group', dfs = filtered_dfs)
    technique_features_df = _combine_features (object= 'technique', dfs = filtered_dfs)
    interaction_matrix = _make_interaction_matrix(
        user_IDs_df= filtered_dfs['groups_df'],
        item_IDs_df= filtered_dfs['techniques_df'],
        positive_cases= filtered_dfs['labels_df']
    )
    if save_as_csv:
        res_dfs = {
            'X_technique' : technique_features_df,
            'X_group' : group_features_df ,
            'y' : interaction_matrix,
        }
        utils.batch_save_df_to_csv (res_dfs, target_path, postfix = 'cleaned', output_list_file= 'cleaned')
    return technique_features_df, group_features_df, interaction_matrix