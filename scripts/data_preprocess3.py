"""
last update: 2023-09-25
data preprocess pipeline V3. Steps:
1. Extract the data from `data/raw/enterprise-attack.json` and save as pands dataframe (`src.data.ingestion2`)
2. Clean the data to achive only the parts that can be used for training (`src.data.cleaning2`)
3. Select the features that will be used for training (`src.data.select_features`)
    - The selected features are defined in a yaml file in `configs/` folder. 
    The file name is one of the arguments when running the script
4. Build the selected features from the previous step
5. Save the results as csv files including
    - The interation matrix between Groups and Techniques
    - Built Group features
    - Built Technique features
6. The list of exported files are stored in PREPROCESSED.txt
"""

import sys
import os
sys.path.append("..")
import argparse
import yaml

### MODULES
from src.data.utils import batch_save_df_to_csv
from src.data.constants import GROUP_ID_NAME, TECHNIQUE_ID_NAME, LABEL_NAME

from src.data.ingestion2 import collect_data
from src.data.cleaning2 import clean_data
from src.data.select_features import select_features
from src.data.build_features2 import build_features

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
CONFIG_FOLDER = os.path.join (ROOT_FOLDER, 'configs')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('-config', required= True,
                         type=str,
                         help = 'name of config file to preprocess the data')
    parser.add_argument ('--last-only','-lo', type = bool, default= True,help='Option: Do not save the tables for intermediary steps, only save the LAST processed tables')
    args = parser.parse_args()
    last_only = args.last_only
    config_file_name = args.config
    #### SETTING: option to save tables in intermediary steps
    save_intermediary_table = not last_only
    
    #### SETTING: load config file config_file_name
    if not config_file_name.endswith ('.yaml'): config_file_name += '.yaml'
    config_file_path = os.path.join (CONFIG_FOLDER, config_file_name)
    with open (config_file_path, 'r') as config_file:
        config = yaml.safe_load (config_file)
        
    selected_group_features = config['selected_group_features']
    selected_technique_features = config['selected_technique_features']
    
    collect_data ()
    
    #### CLEANING DATA / SELECTING FEATURES
    
    technique_features, group_features, interaction_matrix = clean_data(save_as_csv = save_intermediary_table)
    technique_features, group_features = select_features(technique_features_df= technique_features,
                                                         technique_feature_names= selected_technique_features, 
                                                         group_features_df= group_features,
                                                         group_feature_names=selected_group_features,
                                                         save_as_csv= save_intermediary_table)
    
    #### LAST STEPS (save the output tables as csv)
    
    # BUILD FEATURES FOR INPUT
    ## note: for now all features are one-hot encoded
    technique_features, group_features = build_features (
        technique_features_df = technique_features,
        technique_feature_names = selected_technique_features,
        group_features_df = group_features,
        group_features_names = selected_group_features,
        save_as_csv= save_intermediary_table
    )
    
    dfs ={
        'y_cleaned': interaction_matrix,
        'X_group_onehot': group_features,
        'X_technique_onehot': technique_features,
    }
    batch_save_df_to_csv (file_name_dfs= dfs, target_path=TARGET_PATH, output_list_file = 'PREPROCESSED')
    
    
if __name__ == '__main__':
    main()