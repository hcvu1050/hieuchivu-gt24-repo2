"""
Used to gather the data from MITRE ATT&CK. It extracts the following pandas DataFrames and save as csv files in "data/interim"

technique_df.csv: list of all Techniques
techniques_mitigations_df.csv: list of all Mitigations for each Techniques
groups_df.csv: list of all Groups
groups_techniques_df.csv: list of all Techniques used by each Group
groups_software_df.csv: list of all Software used by each Group
"""

### 2023-09-06 default code, update later

# -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

### end of default code

from stix2 import MemoryStore
import mitreattack.attackToExcel.stixToDf as stixToDf
import pandas as pd

import os
# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
MITRE_ATTCK_FILE_PATH = os.path.join(ROOT_FOLDER, 'data/raw', 'enterprise-attack.json')
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def _save_df_to_csv (path, filename: str, df: pd.DataFrame ):
    """save pandas DataFrame as csv file within specified path

    Args:
        path (string)
        df (pandas DataFrame)
    """
    if not filename.endswith (".csv"): filename+= ".csv"
    
    os.makedirs (path, exist_ok= True)
    output_file = os.path.join(path, filename)
    df.to_csv (output_file, index= False)
    
def _batch_save_df_to_csv (file_name_dfs: dict, path):
    """
    save the DataFrames stored in a dict as csv file. The filenames are the keys in the dict
    """
    for key in file_name_dfs.keys():
        _save_df_to_csv (
            path = path,
            filename = key,
            df = file_name_dfs[key]
        )
    
    print ("files saved to", path)
    
    for key in file_name_dfs.keys():
        print (key, ".csv", sep = '')

def read_data_local(file_path = MITRE_ATTCK_FILE_PATH):
    """
    v1.0
    
    Reads local file 'enterprise-attack.json' from `path` (default = "data/raw").
    Returns the following DataFrames:
        techniques_df, \n
        techniques_mitigations_df, \n
        groups_df, \n
        groups_techniques_df, \n
        groups_software_df\n
    """
    
    attackdata = MemoryStore ()
    attackdata.load_from_file (file_path)
    techniques_data = stixToDf.techniquesToDf(attackdata, "enterprise-attack")
    groups_data = stixToDf.groupsToDf (attackdata)
    
    techniques_df = techniques_data["techniques"]
    techniques_mitigations_df = techniques_data['associated mitigations']
    groups_df = groups_data['groups']
    groups_techniques_df = groups_data['techniques used']
    groups_software_df = groups_data['associated software']
        
    return techniques_df, techniques_mitigations_df, groups_df, groups_techniques_df, groups_software_df

def collect_data(target_path = TARGET_PATH):
    """
    v1.0
    save the following DataFrames as csv in specifed path (default = "data/interim"):
        techniques_df, \n
        techniques_mitigations_df, \n
        groups_df, \n
        groups_techniques_df, \n
        groups_software_df\n
    
    """
    techniques_df, techniques_mitigations_df, groups_df, groups_techniques_df, groups_software_df = read_data_local()
    
    dfs = {
    "collected_techniques_df" : techniques_df,
    "collected_techniques_mitigations_df" : techniques_mitigations_df,
    "collected_groups_df": groups_df,
    "collected_groups_techniques_df" : groups_techniques_df,
    "collected_groups_software_df" : groups_software_df,
    }
    _batch_save_df_to_csv (dfs, target_path)
    return techniques_df, techniques_mitigations_df, groups_df, groups_techniques_df, groups_software_df
    
    