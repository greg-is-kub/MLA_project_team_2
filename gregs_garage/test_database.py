import pandas as pd
import numpy as np


if __name__ == "__main__" :
    load_path = "DB.csv"
    col_name = ["covided" , "sane"]
    DB_df = pd.read_csv( filepath_or_buffer = load_path )#,dtype = int )#, names = col_name)
    DB = pd.Series( dtype = np.float64 )
    for i in DB_df.keys() :
        if i != "0" :
            DB[i] = DB_df[i].squeeze()
    key = DB.keys()
    
    for i in key :
        print(i)
        print(DB[i] , end = "\n\n\n")
    