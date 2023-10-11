"""
last update: 2023-09-21
Build features for training the model.
Current functions inlude:
1. One-hot encoding for single-valued string. (for example, "MacOS", Windows)
2. One-hot encoding for multiple-valued strings (for example, "MacOS, Windows") and 
    handling variations of the same value: 
        uppercase/lowercase ("MacOS" is similar to "macos"), 
        hyphens/non-hyphen ("anti-virus" is similar to "anti virus")
"""
import os
import pandas as pd
from . import utils
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to the txt file that contains a list of filtered features (exported from selecfeatures.py)
SOURCE_LIST_FILE = os.path.join (SOURCE_PATH, 'selected_features.txt')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
TECHNIQUE_TABLE_PREFIX = 'X_technique'
GROUP_TABLE_PREFIX = 'X_group'
from ..constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME

def build_features_onehot(technique_features_df: pd.DataFrame|None, 
                   technique_feature_names: list|None,
                   group_features_df: pd.DataFrame|None,
                   group_features_names: list |None,
                   target_path = TARGET_PATH, 
                   save_as_csv = True):
    """One-hot encoding features for Techniques and Group. 
    The features to be one-hot encoded are defined in `technique_feature_names` and `group_feature_names`"""
    print (PROCESS_RUNNING_MSG)
    
    if (technique_features_df is None) or (group_features_df is None):
    ###  If don't receive the tables as args, get the table from files instead
        technique_features_df, group_features_df = _get_data()
    
    onehot_technique_features_df = _onehot_encode_features (technique_features_df, 
                                                         ID = TECHNIQUE_ID_NAME, 
                                                         feature_names= technique_feature_names)
    onehot_group_features_df = _onehot_encode_features (group_features_df,
                                                     ID = GROUP_ID_NAME, 
                                                     feature_names= group_features_names)
    if save_as_csv:
        dfs = {
            TECHNIQUE_TABLE_PREFIX : onehot_technique_features_df,
            GROUP_TABLE_PREFIX: onehot_group_features_df
        }
        utils.batch_save_df_to_csv (dfs, target_path, postfix = 'onehot', output_list_file= 'built_features')
    return onehot_technique_features_df, onehot_group_features_df


def _get_data():
    with open (SOURCE_LIST_FILE, 'r') as file:
        csv_file_names = file.read().splitlines()
        
    technique_file_name = [file_name for file_name in csv_file_names if file_name.startswith(TECHNIQUE_TABLE_PREFIX)]
    technique_features_df = pd.read_csv (os.path.join (SOURCE_PATH, technique_file_name[0]))

    group_file_name = [file_name for file_name in csv_file_names if file_name.startswith(GROUP_TABLE_PREFIX)]
    group_features_df = pd.read_csv (os.path.join (SOURCE_PATH, group_file_name[0]))
    return technique_features_df, group_features_df


def _onehot_encode_features(df: pd.DataFrame, ID: str, feature_names: list, feature_sep_char = ',') -> pd.DataFrame():
    """Build one-hot encoded features in table `df` for the columns indicated by `feature_names`.\n
    Returns the entire DataFrame with the specified feature one-hot encoded.\n
    Work for 2 cases\n
    (1): Single-valued strings (e.g.: "MacOS" , "Windows")\n
    (2): Multiple-valued strings (e.g.: "MacOS, Windows"). The default char that separates the values is `,`
    """
    # get the columns that will not change
    constant_names = [col for col in df.columns if col not in feature_names]
    constant_cols = df[constant_names]

    onehot_feature_dfs = []
    for feature_name in feature_names:
        # check if the features are single valued strings
        multi_valued = df[feature_name].str.contains(feature_sep_char, case=False).any()
        # single valued feature 
        if not multi_valued:
            feature_onehot = pd.get_dummies (df[feature_name], dtype = float)
        # multiple valued feature
        else:
            feature_onehot = df[feature_name].str.replace (r',\s*', 
                                                           ',', 
                                                           regex = True)
            ## replace occurrences of a comma followed by zero or more whitespace characters (r',\s*') with
            ## a comma (',')
            
            feature_onehot = feature_onehot.str.lower()            
            feature_onehot = feature_onehot.str.replace (r'[-/]', ' ', regex = True)
            
            feature_onehot = feature_onehot.str.get_dummies (sep = ',')
        
        onehot_feature_dfs.append (feature_onehot)
        # combine the one-hot encoded features with the constant columns
    onehot_feature_dfs = [constant_cols] + onehot_feature_dfs
    df_onehot = pd.concat (
        onehot_feature_dfs,
        axis = 1
    )
    
    df_onehot = df_onehot.groupby(ID).max().reset_index()
            
    return df_onehot

