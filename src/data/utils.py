import os

def batch_save_df_to_csv (file_name_dfs: dict, target_path, prefix =''):
    """
    Saves the DataFrames stored in a dict as csv file. 
    file_name_dfs: key = filenames, value = DataFrame
    
    """
    for key in file_name_dfs.keys():
        os.makedirs (target_path, exist_ok = True)
        
        filename = prefix + key
        if not filename.endswith (".csv"): filename+= ".csv"
        output_file = os.path.join(target_path, filename)
        
        df = file_name_dfs[key]
        df.to_csv (output_file, index = False)
    
    print ("Finished: files saved to", target_path)
    
    for key in file_name_dfs.keys():
        print ("\t", prefix + key, ".csv", sep = '')