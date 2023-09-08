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

def align_data(input_df: pd.DataFrame, target_df: pd.DataFrame, object: str):
    """ Aligns the instances in input_df so that they match with their corresponding targets in target_df
    
    Args:
        object (str): "group" or "technique"
    """
    id_name = ''
    if object == 'group':
        id_name = 'group_ID'
    elif object == 'technique':
        id_name = 'technique_ID'
    df_aligned = pd.merge(left = input_df, right= target_df, on = id_name, how = 'right')
    return df_aligned