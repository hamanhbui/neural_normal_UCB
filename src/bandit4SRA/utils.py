import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataframe = pd.read_csv('data/out2.csv')
list_tmp = []
for index in range(1, dataframe.shape[1], 1): 
	for t in range(1):    
		columnSeriesObj = dataframe.iloc[:, index]
		list_tmp.append(columnSeriesObj.values)

list_data = []
for i in range(len(list_tmp)):
	list_data.append([])
	for j in range(len(list_tmp[i])):
		tmp = list_tmp[i][j].strip('][').split(', ')
		list_data[i].append(tmp)

list_data = np.array(list_data, dtype=float)
list_tmp = list_data.reshape(list_data.shape[0], -1)
mean = np.mean(list_tmp, axis=0)
cov = np.cov(list_tmp, rowvar=0)

T = list_data.shape[0]
K = list_data.shape[1]
print(T)
print(K)
quit()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataframe = pd.read_csv('data/covid_data2/hhs_data_cleaned.csv')

list_out, list_tmp = [], []
for i in range(len(dataframe)):
    day = dataframe.loc[i, 'date']
    context = [dataframe.loc[i, 'census'], dataframe.loc[i, 'capacity_icu'], dataframe.loc[i, 'census_icu'],
        dataframe.loc[i, 'census_covid'], dataframe.loc[i, 'census_covid_icu'], dataframe.loc[i, 'admissions_covid'],
        dataframe.loc[i, 'capacity']]
    list_tmp.append([day, context])
    if i != len(dataframe)-1:
        if dataframe.loc[i, 'hospital_id'] != dataframe.loc[i+1, 'hospital_id']:
            list_out.append([dataframe.loc[i, 'hospital_id'], list_tmp])
            list_tmp = []
    else:
        list_out.append([dataframe.loc[i, 'hospital_id'], list_tmp])

dict_cols = {"TSA AREA": [list_out[0][0]]}
for i in range(len(list_out[0][1])):
    dict_cols[list_out[0][1][i][0]] = [list_out[0][1][i][1]]
    
for i in range(1, len(list_out)):
    dict_cols["TSA AREA"].append(list_out[i][0])
    for j in range(len(list_out[i][1])):
        dict_cols[list_out[i][1][j][0]].append(list_out[i][1][j][1])
    
df = pd.DataFrame(dict_cols)
df.to_csv('data/out.csv', index=False)