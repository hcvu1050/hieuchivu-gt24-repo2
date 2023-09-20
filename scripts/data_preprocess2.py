import sys
import os
sys.path.append("..")
import argparse
from src.data.ingestion2 import collect_data
from src.data.cleaning2 import clean_data
from src.data.select_features import select_features
TRAIN_CV_TEST_RATIO = [.7,.15, .15]

def main():
    collect_data ()
    technique_features, group_features, interaction_matrix = clean_data(save_as_csv = True)
    technique_features, group_features = select_features(technique_features= ['platforms', 'defenses_bypassed'], group_features=['software_ID'])


if __name__ == '__main__':
    main()