"""
last update: 2023-09-09
- used to split the dataset into train, cross-validation, and test set. The ratio of the sets are defined by the users
- Currently there are two ways to split the dataset:
	1. Split the dataset randomly into train, cross-validation, and test set
	2. Split the dataset by randomly split the Groups. Then, based on the split Groups, build the train, cross-validation, and test set. 
        With this process, data from a Group and be only from either train, or cross-validation, or test set.
"""
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

def split_data (df: pd.DataFrame, ratio: list, save_as_csv = True):
    """ Splits data randomly based on ratio.
    # Future task: ratio should be added up to 1
    """
    train_size, cv_size, test_size =  ratio [0], ratio[1], ratio[2]
    # split train set
    train_df, test_df = train_test_split(df, train_size = train_size, random_state=RANDOM_STATE)
    # relative ratio for cv_size/ test_size
    rel_cv_size = cv_size / (cv_size + test_size)
    rel_test_size = 1 - rel_cv_size
    # split cv and test set
    cv_df, test_df = train_test_split(test_df, test_size= rel_test_size, random_state= RANDOM_STATE)
    if save_as_csv:
        dfs = {
            'train_df': train_df,
            'cv_df': cv_df,
            'test_df': test_df
        }
        utils.batch_save_df_to_csv (dfs, target_path = TARGET_PATH , prefix= 'random_splitted_')
    
    return train_df, cv_df, test_df

def split_data_by_group (df: pd.DataFrame, ratio: list, save_as_csv = True):
    """ Splits data by group randomly so that: data of a group ONLY belong to either train set, or cv set, or test set.
    """
    # split the unique values of group_ID
    train_size, cv_size, test_size =  ratio [0], ratio[1], ratio[2]
    # split train ids
    train_IDs, test_IDs = train_test_split (df['group_ID'].unique(), train_size=train_size, random_state=RANDOM_STATE)
    # relative ratio for cv_size/ test_size
    rel_cv_size = cv_size / (cv_size + test_size)
    rel_test_size = 1 - rel_cv_size    
    # split cv and test ids
    cv_IDs, test_IDs = train_test_split (test_IDs, test_size= rel_test_size, random_state= RANDOM_STATE)
    # build train, cv, and test sets based on splitted ids
    train_df   = df[df['group_ID'].isin(train_IDs)]
    cv_df      = df[df['group_ID'].isin(cv_IDs)]
    test_df    = df[df['group_ID'].isin(test_IDs)]
    if save_as_csv:
        dfs = {
            'train_df': train_df,
            'cv_df': cv_df,
            'test_df': test_df
        }
        utils.batch_save_df_to_csv (dfs, target_path = TARGET_PATH , prefix= 'splitted_by_group_')
    return train_df, cv_df, test_df

    
    