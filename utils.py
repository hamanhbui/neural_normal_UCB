import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def filter_RA():
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

def filter_ICU():
    dataframe = pd.read_csv('data/tx_tsa_hospitalizations.csv')
    list_out, list_tmp = [], []
    for i in range(len(dataframe)):
        day = dataframe.loc[i, 'date']
        context = [int(dataframe.loc[i, 'patients_covid_total']), int(dataframe.loc[i, 'patients_covid_ward']), int(dataframe.loc[i, 'patients_covid_icu']),
            int(dataframe.loc[i, 'patients_all_total']), int(dataframe.loc[i, 'beds_avail_icu'])]
        list_tmp.append([day, context])
        if i != len(dataframe)-1:
            if dataframe.loc[i, 'tsa_name'] != dataframe.loc[i+1, 'tsa_name']:
                list_out.append([dataframe.loc[i, 'tsa_name'], list_tmp])
                list_tmp = []
        else:
            list_out.append([dataframe.loc[i, 'tsa_name'], list_tmp])
    
    dict_cols = {"TSA AREA": [list_out[0][0]]}
    for i in range(len(list_out[0][1])):
        dict_cols[list_out[0][1][i][0]] = [list_out[0][1][i][1]]
    
    for i in range(1, len(list_out)):
        dict_cols["TSA AREA"].append(list_out[i][0])
        for j in range(len(list_out[i][1])):
            dict_cols[list_out[i][1][j][0]].append(list_out[i][1][j][1])
    
    df = pd.DataFrame(dict_cols)
    df.to_csv('data/out.csv', index=False)

# filter_ICU()


dataframe1 = pd.read_excel('data/texas_ICU_beds.xlsx')

list_data = []
for index in range(2, dataframe1.shape[1], 1): 
	for t in range(1):    
		columnSeriesObj = dataframe1.iloc[:, index]
		list_data.append(columnSeriesObj.values[:-1])

list_data = np.array(list_data, dtype=int)
