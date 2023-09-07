"""


"""
import os
import pandas as pd

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
TECHNIQUE_FEATURES_PATH = os.path.join (ROOT_FOLDER, 'cleaned_technique_features_df.csv')
GROUP_FEATURES_PATH = os.path.join (ROOT_FOLDER, 'cleaned_group_features_df.csv')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

TARGET_PREFIX = 'cleaned_'


def _onehot_encode_features(df: pd.DataFrame, ID: str, feature_names: list, feature_sep_char = ','):
    # get the columns that will not change
    constant_names = [col for col in df.columns if col not in feature_names]
    constant_cols = df[constant_names]


    df_onehot = constant_cols
    for feature_name in feature_names:
        # check if the features are single valued strings
        multi_valued = df[feature_name].str.contains(feature_sep_char, case=False).any()
        # single valued feature 
        if not multi_valued:
            feature_onehot = pd.get_dummies (df[feature_name], dtype = float)
        else: 
            feature_onehot = df[feature_name].str.replace (r',\s*', ',', regex = True)
            feature_onehot = feature_onehot.str.get_dummies (sep = ',')
            
        df_onehot = pd.concat (
            [df_onehot, feature_onehot],
            axis = 1
        )
        df_onehot = df_onehot.groupby(ID).max().reset_index()
            
    # multiple valued feature
    return df_onehot


def build_features():
    technique_features_df = pd.read_csv (TECHNIQUE_FEATURES_PATH)
    group_features_df = pd.read_csv (GROUP_FEATURES_PATH)
    
    onehot_technique_features_df = _onehot_encode_features (technique_features_df, 
                                                         ID = 'technique_ID', 
                                                         feature_names= ['platforms', 'mitigation_ID'])
    onehot_group_features_df = _onehot_encode_features (group_features_df,
                                                     ID = 'group_ID', 
                                                     feature_names= ['software_ID'])
    dfs = {
        'onehot_technique_features_df' : onehot_technique_features_df,
        'onehot_group_features_df': onehot_group_features_df
    }
    
    