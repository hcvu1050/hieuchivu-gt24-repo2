import sys
import os
sys.path.append("..")
import argparse
from src.data.ingestion2 import collect_data
from src.data.cleaning2 import clean_data
from src.data.select_features import select_features
from src.data.build_features2 import build_features
TRAIN_CV_TEST_RATIO = [.7,.15, .15]

def main():
    collect_data ()
    t_selected_features = ['defenses_bypassed', 'software_ID']
    technique_features, group_features, interaction_matrix = clean_data(save_as_csv = True)
    technique_features, group_features = select_features(technique_feature_names= t_selected_features, group_feature_names=['software_ID'])
    technique_features, group_features = build_features (
        technique_features_df = technique_features,
        technique_feature_names = t_selected_features,
        group_features_df = group_features,
        group_features_names = ['software_ID']
    )

if __name__ == '__main__':
    main()