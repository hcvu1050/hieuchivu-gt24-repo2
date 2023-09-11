import sys
import os
sys.path.append("..")
import argparse
from src.data.ingestion import collect_data
from src.data.cleaning import clean_data
from src.data.build_features import build_features
from src.data.splitting import split_data_by_group
from src.data.balancing import naive_random_oversampling
from src.data.aligning import align_input_to_target
from src.data.utils import batch_save_df_to_csv

TRAIN_CV_TEST_RATIO = [.7,.15, .15]

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    parser = argparse.ArgumentParser (description= 'command-line arguments when running {}'.format ('__file__'))
    parser.add_argument ('--last-only','-lo', type = bool, default= True,help='Option: Do not save the tables for intermediary steps, only save the last processed tables')
    args = parser.parse_args()
    last_only = args.last_only
    # option to save tables in intermediary steps
    save_intermediary_table = not last_only
    
    collect_data()
    technique_features, group_features, interaction_matrix = clean_data(save_as_csv = save_intermediary_table)
    technique_features, group_features = build_features(technique_features_df= technique_features,
                                                        group_features_df= group_features,
                                                        save_as_csv = save_intermediary_table)
    
    train_target_df, cv_target_df, test_target_df = split_data_by_group (interaction_matrix, ratio= TRAIN_CV_TEST_RATIO, save_as_csv = save_intermediary_table)
    
    balanced_train_target_df = naive_random_oversampling (train_target_df, save_as_csv= save_intermediary_table)
    
    ### CREATING FINAL TABLES ###
    # aligining: train input
    train_technique_input = align_input_to_target ( feature_df= technique_features,
                                                   object= 'technique',
                                                   target_df= balanced_train_target_df,
                                                   from_set = 'train', save_to_csv= save_intermediary_table)
    train_group_input = align_input_to_target ( feature_df= group_features,
                                                   object= 'group',
                                                   target_df= balanced_train_target_df,
                                                   from_set = 'train', save_to_csv= save_intermediary_table)
    
    
    # aligning: cv input
    cv_technique_input = align_input_to_target (feature_df= technique_features,
                                                  object= 'technique',
                                                  target_df = cv_target_df,
                                                  from_set= 'cv', save_to_csv= save_intermediary_table)
    cv_group_input = align_input_to_target (feature_df= group_features,
                                                  object= 'group',
                                                  target_df = cv_target_df,
                                                  from_set = 'cv', save_to_csv=save_intermediary_table)
    
    # aligning: test input
    test_technique_input = align_input_to_target (feature_df= technique_features,
                                                  object= 'technique',
                                                  target_df=test_target_df,
                                                  from_set= 'test', save_to_csv= save_intermediary_table)
    test_group_input = align_input_to_target (feature_df= group_features,
                                                  object= 'group',
                                                  target_df=test_target_df,
                                                  from_set= 'test', save_to_csv= save_intermediary_table)
    
    ### SAVING FINAL TABLES 
    dfs = {
        'train_y_balanced':             balanced_train_target_df,
        'train_X_technique_aligned':    train_technique_input,
        'train_X_group_aligned':        train_group_input,
        'cv_y':                         cv_target_df,
        'cv_X_technique_aligned':       cv_technique_input,
        'cv_X_group_aligned':           cv_group_input,
        'test_y':                       test_target_df,
        'test_X_technique_aligned':       test_technique_input,
        'test_X_group_aligned':           test_group_input,
    }
    batch_save_df_to_csv (file_name_dfs= dfs, target_path=TARGET_PATH, prefix = 'FINAL')
    
if __name__ == '__main__':
    main()
