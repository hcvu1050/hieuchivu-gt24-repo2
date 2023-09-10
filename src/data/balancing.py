import os
import pandas as pd
from . import utils
from imblearn.over_sampling import RandomOverSampler

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')

# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
# TARGET_PREFIX = 'balanced_'
PROCESS_RUNNING_MSG = "--runing {}".format(__name__)
RANDOM_STATE = 13

def naive_random_oversampling (df: pd.DataFrame, save_as_csv = True):
    print (PROCESS_RUNNING_MSG)
    postfix = 'naive_oversampled'
    X = df[['group_ID','technique_ID']]
    y = df[['target']]
    ros = RandomOverSampler(random_state= RANDOM_STATE)
    x_resampled,y_resampled = ros.fit_resample(X , y)
    res_df = pd.concat ([x_resampled,y_resampled], axis =1)
    if save_as_csv:
        utils.save_df_to_csv (res_df, TARGET_PATH, filename= 'train_y', postfix = postfix)
    return res_df
