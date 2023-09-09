import os
import pandas as pd
from . import utils

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
TARGET_PREFIX = 'splitted_'

RANDOM_STATE = 13

def align_data(input_df: pd.DataFrame, target_df: pd.DataFrame, object: str, save_to_csv: True):
    """ Aligns the instances in input_df so that they match with their corresponding targets in target_df.
    The main purpose of the function is for the input features (group features and technique features)
    Args:
        object (str): "group" or "technique"
    """
    id_name = ''
    filename = ''
    if object == 'group':
        id_name = 'group_ID'
        filename = 'group_features'
    elif object == 'technique':
        id_name = 'technique_ID'
        filename = 'technique_features'
    df_aligned = pd.merge(left = input_df, right= target_df, on = id_name, how = 'right')
    
    # remove unecessary columns after merging
    if object == 'group':
        df_aligned.drop (columns= ['technique_ID', 'target'], inplace= True)
    elif object == 'technique':
        df_aligned.drop (columns= ['group_ID', 'target'], inplace= True)
    
    if save_to_csv:
        utils.save_df_to_csv (df_aligned, target_path = TARGET_PATH, filename= filename, prefix= TARGET_PREFIX)
    return df_aligned