"""
last update: 2023-10-09 Option to oversample train set
- Model1 preprocessing steps (srcript version 1.2)
- Usage: data preprocess pipeline specific to model 1. ❗Only works after running `data_preprocess3`. Steps: 
    1. Load the data exported by running `data_preprocess3`
    2. Split the data into train, train-cv, cv, and test set with ratios defined by a yaml file in `configs/folder`
    3. (Optional step) Oversampling train and train-cv data
    4. Aligning features to labels
    5. Create tensorflow Datasets for train and train-cv sets.
    6. Save the preprocessed data to `data/preprocessed/model1`
- Args: 
    - `config`: name of the `yaml` file in `configs/` that will be used to define the ratios for splitting the data sets.
    - `-lo`  (means 'last only', defaul = `True`): optional argument to save the intermediary data while going through the preprocessing steps.
"""
import sys, os, yaml, argparse
sys.path.append("..")

from src.models.model1.model_preprocess import *
from src.constants import TRAIN_DATASET_FILENAME, TRAIN_CV_DATASET_FILENAME, CV_DATASET_FILENAME, TEST_DATASET_FILENAME
from src.data.utils import batch_save_df_to_csv

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
TARGET_PATH = os.path.join (ROOT_FOLDER, 'data/processed/model1')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required = True,
                         type = str, 
                         help='name of config file preprocess data for model1')
    
    parser.add_argument ('--last-only','-lo', required=True, choices= ['True', 'False'],
                         help='Option to not save the tables for intermediary steps, only save the last processed tables value is either "True" or "False"')
    
    args = parser.parse_args()
    config_file_name = args.config
    last_only = args.last_only
    
    if last_only == "True": last_only = True
    elif last_only == "False": last_only = False
    save_intermediary_table = not last_only
    
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
        
    #### LOAD CONFIGS FROM CONFIG FILE
    if not config_file_name.endswith ('.yaml'): config_file_name += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER,config_file_name)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
    formatted_text = yaml.dump(config, default_flow_style=False, indent=2, sort_keys=False)
    print ('---config for model 1 preprocessing:\n',formatted_text)
    
    data_split = config['data_split']
    resampling = config['resampling']
    train_size, train_cv_size, cv_size, test_size = data_split
    
    #### 1- LOAD DATA
    group_features_df, technique_features_df, labels_df = get_data()
    
    #### 2- SPLIT LABELS
    print ('--splitting data')
    train_y_df, remain_y_df  = split_by_group (labels_df, ratio = train_size)
    train_cv_y_df, remain_y_df = split_by_group (remain_y_df, 
                                                 ratio = train_cv_size/ (train_cv_size + cv_size + test_size))
    cv_y_df, test_y_df = split_by_group (remain_y_df, 
                                          ratio = cv_size/(cv_size + test_size))
    if save_intermediary_table:
        dfs = {
            'train_y': train_y_df,
            'train_cv_y': train_cv_y_df,
            'cv_y': cv_y_df,
            'test_y': test_y_df
        }
        batch_save_df_to_csv (dfs, TARGET_PATH, postfix= 'split')
    
    #### 3- (OPTIONAL) OVERSAMPLING train and train_cv, if train_cv size is set to 0, return an empty dataframe

    if resampling is not None: 
        print ('--resampling data')
        
        train_y_df = label_resample (train_y_df, sampling_strategy= resampling)
        if train_cv_size != 0:
            train_cv_y_df = label_resample (train_cv_y_df, sampling_strategy= resampling)
        if save_intermediary_table:
            dfs = {
                'train_y': train_y_df,
                'train_cv_y': train_cv_y_df,
            }
            batch_save_df_to_csv (dfs, TARGET_PATH, postfix='resampled')
    
    #### 4- ALIGNING features to labels
    ## train set
    print ('--aligning data')
    train_X_group_df = align_input_to_labels (group_features_df, 
                                              object= 'group', 
                                              label_df= train_y_df)
    train_X_technique_df = align_input_to_labels (technique_features_df, 
                                                  object= 'technique', 
                                                  label_df= train_y_df)
    # train_cv set
    train_cv_X_group_df = pd.DataFrame()
    train_cv_X_technique_df = pd.DataFrame()
    if train_cv_size != 0:
        train_cv_X_group_df = align_input_to_labels (group_features_df, 
                                                object= 'group', 
                                                label_df= train_cv_y_df)
        train_cv_X_technique_df = align_input_to_labels (technique_features_df, 
                                                    object= 'technique', 
                                                    label_df= train_cv_y_df)
    # cv set
    cv_X_group_df = align_input_to_labels (group_features_df, 
                                           object= 'group', 
                                           label_df= cv_y_df)
    cv_X_technique_df = align_input_to_labels (technique_features_df, 
                                               object= 'technique', 
                                               label_df= cv_y_df)
    # test set
    test_X_group_df = align_input_to_labels (group_features_df, 
                                             object= 'group', 
                                             label_df= test_y_df)
    test_X_technique_df = align_input_to_labels (technique_features_df, 
                                               object= 'technique', 
                                               label_df= test_y_df)
    
    if save_intermediary_table:
        dfs = {
        'train_X_group':        train_X_group_df,
        'train_X_technique':    train_X_technique_df,
        'train_cv_X_group':     train_cv_X_group_df,
        'train_cv_X_technique': train_cv_X_technique_df,
        'cv_X_group':           cv_X_group_df,
        'cv_X_technique':       cv_X_technique_df,
        'test_X_group':         test_X_group_df,
        'test_X_technique':     test_X_technique_df,
        }
        batch_save_df_to_csv (dfs, TARGET_PATH, postfix= 'aligned')
        
    #### 5- Make tensor flow datasets
    print ('--building datasets')
    
    train_dataset = build_dataset(X_group_df =      train_X_group_df, 
                                  X_technique_df =  train_X_technique_df,
                                  y_df =            train_y_df, ragged_input= False)
    
    if train_cv_size != 0:
        train_cv_dataset = build_dataset(X_group_df=    train_cv_X_group_df, 
                                        X_technique_df= train_cv_X_technique_df,
                                        y_df=           train_cv_y_df, ragged_input= False)
        
    cv_dataset = build_dataset(X_group_df =         cv_X_group_df, 
                                  X_technique_df =  cv_X_technique_df,
                                  y_df =            cv_y_df, ragged_input= False)
    
    test_dataset = build_dataset(X_group_df =       test_X_group_df, 
                                  X_technique_df =  test_X_technique_df,
                                  y_df =            test_y_df, ragged_input= False)
    
    
    save_dataset (train_dataset, TARGET_PATH, TRAIN_DATASET_FILENAME)
    if train_cv_size !=0:
        save_dataset (train_cv_dataset, TARGET_PATH, TRAIN_CV_DATASET_FILENAME)
    save_dataset (cv_dataset, TARGET_PATH, CV_DATASET_FILENAME)
    save_dataset (test_dataset, TARGET_PATH, TEST_DATASET_FILENAME)
if __name__ == '__main__':
    main()