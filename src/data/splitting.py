import os
import pandas as pd
from . import utils
### CONFIGURATION ###

# Get the root directory of the project
ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))

# path to get collected data
SOURCE_PATH = os.path.join (ROOT_FOLDER, 'data/interim')
# path to save the cleaned data
TARGET_PATH = os.path.join(ROOT_FOLDER, 'data/interim')

