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
TRAIN_CV_TEST_RATIO = [.7,.15, .15]

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

def main():
    collect_data()
    technique_features, group_features, interaction_matrix = clean_data()
    # technique_features, group_features = build_features()
    
    # train_target_df, cv_target_df, test_target_df = split_data_by_group (interaction_matrix, ratio= TRAIN_CV_TEST_RATIO)
    
    # # aligining: train input
    # balanced_train_target_df = naive_random_oversampling (train_target_df)
    # train_technique_input = align_input_to_target ( feature_df= technique_features,
    #                                                object= 'technique',
    #                                                target_df= balanced_train_target_df,
    #                                                from_set = 'train')
    # train_group_input = align_input_to_target ( feature_df= group_features,
    #                                                object= 'group',
    #                                                target_df= balanced_train_target_df,
    #                                                from_set = 'train')
    
    
    # # aligning: cv input
    # cv_technique_input = align_input_to_target (feature_df= technique_features,
    #                                               object= 'technique',
    #                                               target_df = cv_target_df,
    #                                               from_set= 'cv')
    # cv_group_input = align_input_to_target (feature_df= group_features,
    #                                               object= 'group',
    #                                               target_df = cv_target_df,
    #                                               from_set = 'cv')
    
    # # aligning: test input
    # test_technique_input = align_input_to_target (feature_df= technique_features,
    #                                               object= 'technique',
    #                                               target_df=test_target_df,
    #                                               from_set= 'test')
    # test_group_input = align_input_to_target (feature_df= group_features,
    #                                               object= 'group',
    #                                               target_df=test_target_df,
    #                                               from_set= 'test')
    
if __name__ == '__main__':
    main()
