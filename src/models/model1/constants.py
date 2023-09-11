import os

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
INPUT_GROUP_LAYER_NAME = 'input_Group'
INPUT_TECHNIQUE_LAYER_NAME = 'input_Technique'

TRAIN_DATASET_FILENAME = 'train_dataset'
CV_DATASET_FILENAME = 'cv_dataset'
TEST_DATASET_FILENAME = 'test_dataset'

RANDOM_STATE = 13