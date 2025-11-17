import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path  
import os
from rapidfuzz import fuzz, process
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def create_VI(output_folder, year, country, choice,main_csv_save):
    output_cverank=f"{output_folder}/{year}/covrank/covrank_{year}_{country}_{choice}.csv"
    data1 = pd.read_csv(output_cverank)
    _, ext = os.path.splitext(main_csv_save)
    ext = ext.lower()

    if ext in [".xls", ".xlsx"]:
        data2 = pd.read_excel(main_csv_save)
    elif ext == ".csv":
        data2 = pd.read_csv(main_csv_save)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    a=data1.columns
    b=data2.columns
    matches = set(a) & set(b)

    # Find remaining names
    remaining_list1 = set(a) - matches
    remaining_list2 = set(b) - matches
    weight_normal = dict(zip(data1['model'], data1['rank1']))
    weight_quartile = dict(zip(data1['model'], data1['rank']))
    matching_normal = [col for col in data2.columns if col in weight_normal]

    # Multiply the matching columns by the dictionary values and sum across rows
    data2['sum_normal'] = data2[matching_normal].mul(pd.Series(weight_normal), axis=1).sum(axis=1)
    dict_normal = sum(weight_normal.values())

    # Divide the 'sum_column' by the sum of the dictionary values
    data2[f'VI_{choice}_Rank_final'] = data2['sum_normal'] / dict_normal
    if data2[f'VI_{choice}_Rank_final'].isna().all(): data2[f'VI_{choice}_Rank_final'] = 0
    matching_quartile = [col for col in data2.columns if col in weight_quartile]
    # Multiply the matching columns by the dictionary values and sum across rows
    data2['sum_quartile'] = data2[matching_quartile].mul(pd.Series(weight_quartile), axis=1).sum(axis=1)
    dict_quartile = sum(weight_quartile.values())

    # Divide the 'sum_column' by the sum of the dictionary values
    data2[f'VI_{choice}_Quartile'] = data2['sum_quartile'] / dict_quartile
    if data2[f'VI_{choice}_Quartile'].isna().all(): data2[f'VI_{choice}_Quartile'] = 0


    os.makedirs(os.path.join(output_folder, str(year),str("VI")), exist_ok=True)
    VI_output_path = os.path.join(output_folder,year,f"VI",f"FinalVI_{year}_{country}_{choice}.xlsx")
    data2.to_excel(VI_output_path, index=False)
    # print("All filels are save at: ",VI_output_path )
    # print("---------------------------------------------------------------------------------")
    return VI_output_path  
