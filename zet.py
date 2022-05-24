import pandas as pd




data = pd.read_pickle("../data_file_w_ts_kat_al.pkl")
target = pd.read_pickle("../target_file_w_ts_kat_al.pkl")
print("start")
data.to_csv('data')
print("start")

target.to_csv("target")
print("start")
