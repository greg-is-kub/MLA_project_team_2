import pandas as pd
import numpy as np


if __name__ == "__main__" :
    load_path = "DB.csv"
    col_name = ["covided" , "sane"]
    DB = pd.read_csv( filepath_or_buffer = load_path )#,dtype = int )#, names = col_name)
    
    print(DB.keys() , end="\n\n")
    
#   print( DB[index = 0, 'x_ray'] )
    