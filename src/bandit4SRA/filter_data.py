import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# # df = pd.read_csv('data/capacity_timeseries.csv')
# df = pd.read_csv('data/hospitalization_data.csv')
# # df = pd.read_csv('data/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility.csv')
# # print(df.columns.tolist())
# # print(df.min())
# df_splits = [v for k, v in df.groupby('hospital')]
# start_days, end_days = [], []
# new_df = df_splits[0].copy()

# for idx in range(len(df_splits)):
#     if idx == 0:
#         continue
#     sday = df_splits[idx]['date'].iloc[0]
#     eday = df_splits[idx]['date'].iloc[df_splits[idx].shape[0] - 1]
#     if (sday <= "2020-01-01" and eday >= "2022-12-31"):
#         start_days.append(sday)
#         end_days.append(eday)
#         new_df = pd.concat([new_df, df_splits[idx].copy()], ignore_index=True)

# new_df = new_df.loc[new_df['date'] >= "2020-01-01"]
# new_df = new_df.loc[new_df['date'] <= "2022-12-31"]
# new_df.to_csv('out2.csv', index=False)

dataframe = pd.read_csv('data/test.csv')
list_tmp = []
for index in range(1, dataframe.shape[1], 1): 
	columnSeriesObj = dataframe.iloc[1:, index]
	list_tmp.append(columnSeriesObj.values)

list_data = []
for i in range(len(list_tmp)):
	list_data.append([])
	for j in range(len(list_tmp[i])):
		tmp = list_tmp[i][j].strip('][').split(', ')
		list_data[i].append(tmp)

list_data = np.array(list_data, dtype=float)
maxv = 0
for idx in range(list_data.shape[0]):
	tmp = sum(list_data[idx][:,8])
	if maxv < tmp:
		maxv = tmp

print(list_data[idx][1][8])
print(list_data[idx][:,8])
quit()

df_1 = pd.read_csv('out.csv')
df_2 = pd.read_csv('out2.csv')
df_splits_1 = [v for k, v in df_1.groupby('hospital')]
df_splits_2 = [v for k, v in df_2.groupby('hospital')]

cols = ["TSA AREA"]
for idx in range(len(df_splits_1[0])):
	cols.append(df_splits_1[0]['date'].iloc[idx])

list_raws = [cols]
def fix_nan(value):
	if math.isnan(value) is True:
		return 0
	return value

for idx in range(len(df_splits_1)):
	df_tmp_1 = df_splits_1[idx]
	df_tmp_2 = df_splits_2[idx]
	raw = [df_tmp_1['hospital'].iloc[0]]
	for jdx in range(len(df_tmp_1)):
		raw_tmp = []
		raw_tmp.append(fix_nan(df_tmp_2['admissions_icu'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['admissions_acute'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['admissions_combined'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['admissions_combined_ped'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['active_icu'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['active_acute'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['active_combined'].iloc[jdx]))
		raw_tmp.append(fix_nan(df_tmp_2['active_combined_ped'].iloc[jdx]))
		raw_tmp.append(int(df_tmp_1['beds_combined_adultped'].iloc[jdx]))
		raw.append(raw_tmp)
	if len(raw) == 1097:
		list_raws.append(raw)
# print(list_raws)
# quit()
df = pd.DataFrame(list_raws)
df.to_csv('data/test.csv', index=False)
	