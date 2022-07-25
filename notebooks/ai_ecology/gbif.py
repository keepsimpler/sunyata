# %%
# from pygbif import species as species
# from pygbif import occurrences as occ
# # %%
# sp_list = ['Cyanocitta stelleri', 'Junco hyemalis', 'Aix sponsa',
#           'Ursus americanus', 'Pinus conorta', 'Poa annuus']
# # %%
# sp_keys = [species.name_backbone(sp)['usageKey'] for  sp in sp_list]
# # %%
# occ.search(taxonKey=sp_keys[0], limit=1)
# # %%
# # occ.download('taxonKey = 3119195', user='fengwf', pwd='dyc2302c', email='fengwf@hpu.edu.cn')
# # %%
# occ.search(
#     continent='europe', 
#     country='NL', 
#     basisOfRecord=["HUMAN_OBSERVATION ", "MACHINE_OBSERVATION"], 
#     hasCoordinate=True,
#     hasGeospatialIssue=False,
#     occurrenceStatus='present',
#     fields = 'minimal',
#     limit=1)
# # %%
# occ.download(['continent = europe', 'country = NL', 'basisOfRecord = HUMAN_OBSERVATION', 'hasCoordinate = True', 'hasGeospatialIssue = False', 'occurrenceStatus = present', 'fields = minimal'], user='fengwf', pwd='dyc2302c', email='fengwf@hpu.edu.cn')

# %%  GBIF.org
data_path = '.data/gbif/0390049-210914110416597.csv'  # 0388370-210914110416597

# %%
import pandas as pd
# %%
df = pd.read_csv(data_path, sep='\t', on_bad_lines='warn')
# %%
df.head()
# %%
df.columns
# %%
selected_columns = ['occurrenceID', 'decimalLatitude', 'decimalLongitude', 'species']
# %%
len(df)
# %%
df['taxonRank'].unique()
# %%
df.groupby(['taxonRank'])['taxonRank'].count()
# %%
df[df['taxonRank']!='SPECIES'][['scientificName', 'species']]
# %%
df_selected = df[selected_columns].dropna()
# %%
len(df_selected['species'].unique())
# %%
len(df_selected)
# %%
df_selected.head(n=10)
# %%
len(df_selected['decimalLongitude'].unique())

# %%
from math import cos, sin, asin, sqrt, radians

def calc_distance(lat1: float, lon1: float, lat2: float, lon2:float, earth_radius: float=6371.):
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees)
    using Haversine Formula
    """
    # 1. convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # 2. haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = earth_radius * c
    m = 1000 * km
    return m
# %%
lon = df_selected['decimalLongitude'].unique()
lat = df_selected['decimalLatitude'].unique()
len(lon), len(lat)
# %%
import numpy as np
df_selected.shape
# %%
df_selected.to_csv('.data/gbif/nl.csv')
# %%
df_selected = pd.read_csv('.data/gbif/nl.csv')
# %%
df_selected.dtypes
# %%
df_selected['label'] = df_selected.species.astype('category').cat.codes
# %%
dataset = df_selected[['decimalLatitude', 'decimalLongitude', 'label']].to_numpy()
# %%
dataset.dtype
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
torch_dataset = torch.Tensor(dataset)
# %%
torch_dataset.sort(dim=0)
# %%
values, counts = np.unique(dataset, axis=0, return_counts=True)
values.shape
# %%
lat_sorted = values[np.argsort(values[:,0])]
lat_sorted
# %%
lon_sorted = values[np.argsort(values[:,1])]
lon_sorted

# %%
distances = [list() for i in range(len(lon_sorted))]
# %%
cur = 0
max_meters = 1000
for i, row1 in enumerate(lon_sorted):
    print(i)
    lat1, lon1 = row1[0], row1[1]
    for j, row2 in enumerate(lon_sorted):
        lat2, lon2 = row2[0], row2[1]
        if i != j:
            distance = calc_distance(lat1, lon1, lat2, lon2)
            print(distance)
            if distance < max_meters:
                distances[i].append(j)
# %%
i, row1 = 0, lon_sorted[0]
for j, row2 in enumerate(lon_sorted):
    lat2, lon2 = row2[0], row2[1]
    if i != j:
        distance = calc_distance(lat1, lon1, lat2, lon2)
        print(distance)
# %%
