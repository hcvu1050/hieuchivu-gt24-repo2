import os
import pandas as pd
from . import utils

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the built-feature data
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
TARGET_PREFIX = 'aligned_'

RANDOM_STATE = 13
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)

def align_input_to_target(feature_df: pd.DataFrame, object: str, target_df: pd.DataFrame, from_set: str , save_to_csv = True):
    """ Aligns the instances in feature_df so that they match with their corresponding targets in target_df.
    The main purpose of the function is for the input features (group features and technique features)\n
    Args:
        feature_df (pd.DataFrame): DataFrame containing input's features \n
        target_df (pd.DataFrame): DataFrame that feature_df will be aligned to \n
        from_set (str): ('train'|'cv'|'test') describes which set does target_df come from (train, cross-validation, or test)\n
        object (str): "group" or "technique"\n
    """
    print (PROCESS_RUNNING_MSG)
    id_name = ''
    filename = ''
    if from_set in ('train', 'cv','test'):
        filename = from_set
    if object == 'group':
        id_name = 'group_ID'
        filename += 'group_input'
    elif object == 'technique':
        id_name = 'technique_ID'
        filename != 'technique_input'
    df_aligned = pd.merge(left = feature_df, right= target_df, on = id_name, how = 'right')
    
    # remove unecessary columns after merging
    if object == 'group':
        df_aligned.drop (columns= ['technique_ID', 'target'], inplace= True)
    elif object == 'technique':
        df_aligned.drop (columns= ['group_ID', 'target'], inplace= True)
    
    if save_to_csv:
        utils.save_df_to_csv (df_aligned, target_path = TARGET_PATH, filename= filename, prefix= TARGET_PREFIX)
    return df_aligned