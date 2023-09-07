"""
used to clean the data by reducing outliers/noise, handling missing values, etc.
"""

import os
import pandas as pd
# Get the root directory of the project
root_folder = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
# path to save the cleaned data
target_path = os.path.join(root_folder, 'data/interim')

techniques_df           = pd.readcsv (os.path.join (root_folder,'data/interim', 'collected_techniques_df.csv'))
mitigations_df          = pd.readcsv (os.path.join (root_folder,'data/interim', 'collected_mitigations_df.csv'))
groups_df               = pd.readcsv (os.path.join (root_folder,'data/interim', 'collected_groups_df.csv'))
groups_techniques_df    = pd.readcsv (os.path.join (root_folder,'data/interim', 'collected_groups_techniques_df.csv'))
groups_software_df      = pd.readcsv (os.path.join (root_folder,'data/interim', 'collected_groups_software_df.csv'))

selected_column_names = {
    # names of the selected columns in the collected data that will be used for training processes
    'techniques_df' : ['ID', 'platforms'],
    'mitigation_df': ['source ID', 'target ID'], 
    'groups_df' : ['ID'],
    'groups_techniques_df': ['source ID', 'target ID'],
    'groups_software_df' : ['source ID', 'target ID']
}

def 