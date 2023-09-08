import os
import pandas as pd
from . import utils
from sklearn.model_selection import train_test_split

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to get cleaned data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the built-feature data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')
TARGET_PREFIX = 'splitted_'

RANDOM_STATE = 13

def data_split (df: pd.DataFrame, train_size, test_size):
    df_train, df_test = train_test_split(df, train_size = train_size, random_state=RANDOM_STATE)
    df_cv, df_test = train_test_split(df, test_size= test_size, random_state= RANDOM_STATE)
    return df_train, df_cv, df_test
    