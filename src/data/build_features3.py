"""
last update: 2023-10-12
build_features version 3
Build features for training the model.
Current functions inlude:
1. One-hot encoding feature
2. Frequency encoding feature
"""
import os
import pandas as pd
import category_encoders as ce
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
                                                         id = TECHNIQUE_ID_NAME, 
                                                         feature_names= technique_feature_names)
    onehot_group_features_df = _onehot_encode_features (group_features_df,
                                                     id = GROUP_ID_NAME, 
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


def _onehot_encode_features(df: pd.DataFrame, id: str, feature_names: list ) -> pd.DataFrame():
    """Build one-hot encoded features in table `df` for the columns indicated by `feature_names`.\n
    Returns the entire DataFrame with the specified feature one-hot encoded.\n
    The values stored in the specified columns MUST be lists of single-valued strings.
    """
    # get the columns that will not change
    constant_col_names = [col for col in df.columns if col not in (feature_names+[id])]
    constant_cols = df[constant_col_names]
    id_col = df[[id]]
    onehot_feature_dfs = []
    for feature_name in feature_names:
        feature_single_valued = df[[feature_name]].explode(feature_name)
        feature_onehot = pd.get_dummies (feature_single_valued[feature_name], dtype= float)
        feature_onehot = feature_onehot.groupby(level=0).max()
        onehot_feature_dfs.append (feature_onehot)
    # ❗AVOID DATA LOSS, only group as max value for one-hot encoded features
    onehot_feature_dfs = [id_col] + onehot_feature_dfs
    onehot_features_df = pd.concat (
        onehot_feature_dfs, 
        axis = 1)
    onehot_features_df = onehot_features_df.groupby(id).max().reset_index()
    
    constant_cols = [id_col] + [constant_cols]
    constant_cols_df = pd.concat (
        constant_cols,
        axis = 1
    )

    res_df = pd.merge (left = constant_cols_df, right= onehot_features_df, on = id, how = 'left')
    return res_df

def _frequency_encode_features (df: pd.DataFrame, id: str, feature_names: list, feature_sep_char = ',') -> pd.DataFrame():
    """Build frequency encoded features in table `df` for the columns indicated by `feature_names`.\n
    Returns the entire DataFrame with the specified feature frequency encoded.\n
    Work for 2 cases\n
    (1): Single-valued strings (e.g.: "MacOS" , "Windows")\n
    (2): Multiple-valued strings (e.g.: "MacOS, Windows"). The default char that separates the values is `,`
    """
    count_enc = ce.CountEncoder(normalize=True)
    # get the columns that will not change
    constant_names = [col for col in df.columns if col not in feature_names]
    constant_cols = df[constant_names]
    freq_encoded_feature_dfs = []
    
    for feature_name in feature_names:
        # check if the features are single valued strings
        multi_valued = df[feature_name].str.contains(feature_sep_char, case=False).any()
        if not multi_valued:
            freq_encoded = count_enc.fit_transform (df[feature_name], return_df = True)
            freq_encoded_one_hot = pd.get_dummies (freq_encoded[feature_name], dtype = float)
            freq_encoded_one_hot_true_val = freq_encoded_one_hot.multiply (freq_encoded_one_hot.columns, axis = 1)
        
        # else:
        else:   
            feature = df[feature_name].str.replace(r',\s*',',', regex = True)
            feature = feature.str.lower()
            feature = feature.str.replace (r'[-/]', ' ', regex = True)
            feature = feature.str.split(',')
            feature = feature.explode(ignore_index=False)
            feature_freq_encoded = count_enc.fit_transform(feature, return_df = True)
            feature_freq_encoded_oh = pd.get_dummies (feature_freq_encoded[feature_name], dtype = float)
            # combine one-hot values into one vector by the index (The index is kept when calling feature.explode)
            feature_freq_encoded_oh = feature_freq_encoded_oh.groupby(level=0).max()
            freq_encoded_one_hot_true_val = feature_freq_encoded_oh.multiply(feature_freq_encoded_oh.columns, axis= 1)
        
        freq_encoded_feature_dfs.append (freq_encoded_one_hot_true_val)
    freq_encoded_feature_dfs = [constant_cols] + freq_encoded_feature_dfs
    df_freq_encode = pd.concat(
        freq_encoded_feature_dfs, axis= 1
    )
    # ❗
    df_freq_encode = df_freq_encode.groupby(id).max().reset_index()
    return df_freq_encode