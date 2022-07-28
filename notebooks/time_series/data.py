# %%
data_path = '.data/time_series/exchange_rate.csv'
# %%
import pandas as pd
# %%
df_raw = pd.read_csv(data_path)
# %%
# df_raw.columns: ['date', ...(other features), target feature]
cols = list(df_raw.columns)
target = 'OT'
cols.remove(target)
cols.remove('date')
cols
# %%
num_train = int(len(df_raw) * 0.7)
num_test = int(len(df_raw) * 0.2)
num_vali = len(df_raw) - num_train - num_test
num_train, num_test, num_vali
# %%
seq_len = 24 * 4 * 4
label_len = 24 * 4
pred_len = 24 * 4
# %%
border1 = 0
border2 = num_train
# %%
features = 'M' # or 'S', M means Multivariate, S means Single
# %%
df_data = df_raw[df_raw.columns[1:]]
# %%
train_data = df_data[0:num_train]
# %%
from sklearn.preprocessing import StandardScaler
# %%
scaler = StandardScaler()
# %%
scaler.fit(train_data.values)
# %%
data = scaler.transform(df_data.values)
# %%
df_stamp = df_raw[['date']][0:num_train]
# %%
df_stamp['date'] = pd.to_datetime(df_stamp.date)
# %%
df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
# %%
df_stamp.drop(['date'],1).values
# %%
