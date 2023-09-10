import os
import pandas as pd

def save_df_to_csv (df: pd.DataFrame, target_path, filename, prefix = ''):
    os.makedirs (target_path, exist_ok = True)
    if not prefix.endswith ('_'): prefix += '_'
    filename = prefix + filename
    
    if not filename.endswith (".csv"): filename+= ".csv"
    output_file = os.path.join(target_path, filename)
    df.to_csv (output_file, index = False)
    print ("Finished: files saved to", target_path)
    print ("\t", filename, sep = '')

def batch_save_df_to_csv (file_name_dfs: dict, target_path, prefix =''):
    """
    Saves the DataFrames stored in a dict as csv file. \n
    file_name_dfs: \n
        key: filename\n
        value = DataFrame\n
    prefix: a string added before filename
    """
    if not prefix.endswith ('_'): prefix += '_'
    
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key
        if not filename.endswith (".csv"): filename+= ".csv"
        output_file = os.path.join(target_path, filename)
        
        df = file_name_dfs[key]
        df.to_csv (output_file, index = False)
        print ("Saved:\t",  filename, sep = '')
    
    print ("Finished: files saved to", target_path)
    