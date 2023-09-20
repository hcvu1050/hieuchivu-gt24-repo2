"""
select the features from the features created from cleaned2.py
"""
import os
import pandas as pd
from . import utils
from .constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME
TECHNIQUE_TABLE_PREFIX = 'X_technique'
GROUP_TABLE_PREFIX = 'X_group'
RESULT_FILE_POSTFIX = 'selected_features'

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to get collected data
SOURCE_LIST_FILE = os.path.join (SOURCE_PATH, 'cleaned.txt') 
# file that keeps the list of cleaned data

TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)

def _get_data():
    with open (SOURCE_LIST_FILE, 'r') as file:
        csv_file_names = file.read().splitlines()
        
    technique_file_name = [file_name for file_name in csv_file_names if file_name.startswith(TECHNIQUE_TABLE_PREFIX)]
    technique_features_df = pd.read_csv (os.path.join (SOURCE_PATH, technique_file_name[0]))

    group_file_name = [file_name for file_name in csv_file_names if file_name.startswith(GROUP_TABLE_PREFIX)]
    group_features_df = pd.read_csv (os.path.join (SOURCE_PATH, group_file_name[0]))
    # technique_features_df = pd.read_csv (os.path.join (SOURCE_PATH, 'cleaned_technique_features_df.csv'))
    # group_features_df = pd.read_csv (os.path.join (SOURCE_PATH, 'cleaned_group_features_df.csv'))
    return technique_features_df, group_features_df

def select_features(group_features: list = None, technique_features: list = None, save_as_csv = True):
    technique_features_df, group_features_df = _get_data()
    if group_features:
        group_features = [GROUP_ID_NAME] + group_features
        group_features_df = group_features_df[group_features]
    if technique_features: 
        technique_features = [TECHNIQUE_ID_NAME] + technique_features
        technique_features_df = technique_features_df[technique_features]
    if save_as_csv:
        dfs = {
            GROUP_TABLE_PREFIX: group_features_df,
            TECHNIQUE_TABLE_PREFIX: technique_features_df
        }
        utils.batch_save_df_to_csv (dfs, target_path= TARGET_PATH, postfix= RESULT_FILE_POSTFIX, output_list_file = RESULT_FILE_POSTFIX)
    return technique_features_df, group_features_df
