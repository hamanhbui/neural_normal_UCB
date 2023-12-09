import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# dataframe = pd.read_csv('data/dynamic_user_RA/2015_03_02.csv')

# list_filter = ['Bldg3AP91', 'Bldg3AP92', 'Bldg3AP94', 'Bldg3AP95']
# list_time, list_client, list_AP = [], [], []
# for index, row in dataframe.iterrows():
#     if row['AP'] in list_filter:
#         list_time.append(row['timestamp'])
#         list_client.append(row['client'])
#         list_AP.append(row['AP'])

# dict = {'timestamp': list_time, 'client': list_client, 'AP': list_AP} 
# df = pd.DataFrame(dict)
# df.to_csv('data/dynamic_user_RA/out.csv', index=False)

dataframe = pd.read_csv('data/dynamic_user_RA/out.csv')
list_AP = {"Bldg3AP91": 0, "Bldg3AP92": 0, "Bldg3AP94": 0, "Bldg3AP95": 0}
a = 4
list_out = []
for i in range(len(dataframe)):
    list_AP[dataframe.loc[i, 'AP']] += 1
    if i != len(dataframe)-1:
        str_1 = dataframe.loc[i, 'timestamp'].split(':')[0] + ":" + dataframe.loc[i, 'timestamp'].split(':')[1]
        str_2 = dataframe.loc[i+1, 'timestamp'].split(':')[0] + ":" + dataframe.loc[i+1, 'timestamp'].split(':')[1]
        if str_1 != str_2:
            if list_AP["Bldg3AP91"] > a or list_AP["Bldg3AP92"] > a or list_AP["Bldg3AP94"] > a or list_AP["Bldg3AP95"] > a:
                list_out.append([list_AP["Bldg3AP91"], list_AP["Bldg3AP92"], list_AP["Bldg3AP94"], list_AP["Bldg3AP95"]])
            list_AP = {"Bldg3AP91": 0, "Bldg3AP92": 0, "Bldg3AP94": 0, "Bldg3AP95": 0}
    else:
        list_out.append([list_AP["Bldg3AP91"], list_AP["Bldg3AP92"], list_AP["Bldg3AP94"], list_AP["Bldg3AP95"]])

print(list_out)