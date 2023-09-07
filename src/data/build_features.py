"""


"""
import os
import pandas as pd

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

TARGET_PREFIX = 'cleaned_'


def _get_cleaned_filenames():
    csv_files = [filename for filename in os.listdir(SOURCE_PATH) if filename.startswith(TARGET_PREFIX) and filename.endswith('.csv')]
    return csv_files

def _onehot_encode_features(df: pd.DataFrame, ID: str, feature_names: list):
    # single valued feature 
    ## get the columns that will not change
    constant_names = [col for col in df.columns if col not in feature_names]
    constant_cols = df[constant_names]
    
    df_onehot = constant_cols
    for feature_name in feature_names:
        feature_onehot = pd.get_dummies (df[feature_name], dtype = float)
        df_onehot = pd.concat (
            [df_onehot, feature_onehot],
            axis = 1
        )
        df_onehot = df_onehot.groupby(ID).max().reset_index()
    # multiple valued feature
    return df_onehot


def func():
    df = pd.read_csv (os.path.join (SOURCE_PATH, 'cleaned_groups_software_df.csv'))
    df_oh = _onehot_encode_features (df, ID = 'group_ID', feature_names = ['software_ID'])
    return df_oh
    


