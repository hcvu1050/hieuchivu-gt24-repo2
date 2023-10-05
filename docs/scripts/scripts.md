last update: 2023-10-05
# `data_clear_data`
- Usage: delete all files in a subfolder in `data/`. Folder name is specified as an arg. Except for files in `data/raw`, which can NOT be deleted. This script is mainly used for data preprocessing iteration loop.
- Args: 
    - `-dir`,  choices = `'interim'` or `'processed'`.
- Example 
```
python data_clear_data -dir interim
```
deletes all files in `data/interim` folder
# `data_preprocess3`
- Usage: General data preprocess pipeline (script version 3). Steps:
    1. Extract the data from `data/raw/enterprise-attack.json` and save as pands dataframe (`src.data.ingestion2`)
    2. Clean the data to retreive only the portion that can be used for training (`src.data.cleaning2`)
    3. Select the features that will be used for training (`src.data.select_features`)
        - The selected features are defined in a yaml file in `configs/` folder. 
        The file name is one of the arguments when running the script
    4. Build the selected features from the previous step
    5. Save the results as csv files including
        - The interation matrix between Groups and Techniques
        - Built Group features
        - Built Technique features
    6. The list of exported files are stored in `PREPROCESSED.txt`
- Args: 
    - `-config`: name of the `yaml` file in `configs/` folder that will be used to define the selected features for Groups and Techniques.
    - `-lo`  (means 'last only', defaul = `True`): optional argument to save the intermediary data while going through the preprocessing steps.
- Example: 
```
python data_preprocess3 -config data_pp1 -lo False
```
Executes preprocessing steps and select the features based on config file data_pp1. Save all the intermediary data after each preprocessing steps.

# `model1_preprocess`
- Usage: data preprocess pipeline specific to model 1. ❗Only works after running `data_preprocess3`. Steps: 
    1. Load the data exported by running `data_preprocess3`
    2. Split the data into train, train-cv, cv, and test set with ratios defined by a yaml file in `configs/folder`
    3. Oversampling train and train-cv data
    4. Aligning features to labels
    5. Create tensorflow Datasets for train and train-cv sets.
    6. Save the preprocessed data to `data/preprocessed/model1`
- Args: 
    - `config`: name of the `yaml` file in `configs/` that will be used to define the ratios for splitting the data sets.
    - `-lo`  (means 'last only', defaul = `True`): optional argument to save the intermediary data while going through the preprocessing steps.
- Example: 
```
python model1_preprocess -config m1_pp1 -lo False
```
# `model1_train2`
- Usage: Train a single instance of model1 (script version 2). 
- Args: 
    - `-config`:  name of the `yaml` file in `configs/model1/single_train` that will be used to define the hyperparameters for model1
- Example: 
```
python model1_train2 -config m1_train1
```

# `model1_batch_train`
- Usage: Train multiple instances of model1 with different hyperparameters
- Args: 
    - `configs`: name of the subfolder in `configs/` that contains the `yaml` files for different instances (each instance is defined by a set of hyperparameters) of model1
- Example: 
```
python model1_batch_train model1_batch1
```