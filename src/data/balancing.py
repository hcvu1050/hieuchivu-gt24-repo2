import os
import pandas as pd
from . import utils
from imblearn.over_sampling import RandomOverSampler

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')

# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
TARGET_PREFIX = 'balanced_'

RANDOM_STATE = 13
interaction_matrix_df = pd.read_csv (os.path.join (SOURCE_PATH, 'cleaned_interaction_matrix.csv'))

def naive_random_oversampling (df: pd.DataFrame, save_as_csv = True):
    prefix = 'naive_random_oversampled_'
    X = df[['group_ID','technique_ID']]
    y = df[['target']]
    ros = RandomOverSampler(random_state= RANDOM_STATE)
    x_resampled,y_resampled = ros.fit_resample(X , y)
    res_df = pd.concat ([x_resampled,y_resampled], axis =1)
    if save_as_csv:
        utils.save_df_to_csv (res_df, TARGET_PATH, 'interaction_matrix', prefix)
    return res_df
