import pandas as pd
import numpy as np
#from henryCode import bayesian_optimization

def pre_procces_data(file, file_start_rows= 1, file_rows_length = 0, data_cols = None):

    if data_cols != None:
        df = pd.read_excel(file, skiprows=file_start_rows, nrows= file_rows_length, usecols= data_cols)

    else:
        df = pd.read_excel(file, skiprows=file_start_rows, nrows= file_rows_length)

    #print(df)

    dft = df.T
    times = dft.iloc[0]
    temp = dft.iloc[1]
    return print(dft)



pre_procces_data('Example Raw data.xlsx', 10, 93, "A:B")




#print(times)
#print(temp)
#print(dft)