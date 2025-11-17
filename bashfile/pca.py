# import re
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# from sklearn.model_selection import train_test_split
# import argparse
# from pathlib import Path  
# import os
# from rapidfuzz import fuzz, process
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# def pca(output_folder, year, country):
#     vi_path = os.path.join(output_folder, str(year), "VI")
#     pca_path = os.path.join(output_folder, str(year), "PCA")
#     os.makedirs(pca_path, exist_ok=True)

#     expected_choices = ["climate", "conflict", "disease", "socio"]

#     vi_files = {}
#     for fname in os.listdir(vi_path):
#         fl = fname.lower()
#         for choice in expected_choices:
#             if choice in fl and fl.endswith(".xlsx") and country.lower() in fl:
#                 vi_files[choice] = os.path.join(vi_path, fname)

#     missing = [ch for ch in expected_choices if ch not in vi_files]
#     if missing:
#         print(f"⚠️ Missing VI files for: {missing}. PCA will use only available data.")

#     # load available dataframes
#     dfs = {}
#     for choice, path in vi_files.items():
#         try:
#             dfs[choice] = pd.read_excel(path)
#         except Exception as e:
#             print(f"Error reading {path}: {e}")

#     # merge on cluster number (outer)
#     merged_df = None
#     for choice, df in dfs.items():
#         if merged_df is None:
#             merged_df = df.copy()
#         else:
#             merged_df = merged_df.merge(df, on="cluster number", how="outer", suffixes=("", f"_{choice}"))

#     merged_df["proportion_of_low_bmi"] = merged_df["TotlowBmi"].div(merged_df["Totclust"]).replace([np.inf, -np.inf], np.nan)
#     vi_cols = ["VI_climate_Rank_final", "VI_disease_Rank_final",
#            "VI_conflict_Rank_final", "VI_socio_Rank_final"]
#     final_df = merged_df[vi_cols].copy()
#     final_df = final_df.apply(lambda s: s.fillna(s.median()))
#     merged_df[vi_cols] = final_df
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(final_df) 
#     pca = PCA(n_components=1)
#     pc1_scores = pca.fit_transform(X_scaled).ravel()  # shape (n,)
#     merged_df["VI_PCA_raw"] = pc1_scores
#     merged_df["VI_PCA"] = (pc1_scores - pc1_scores.min()) / (pc1_scores.max() - pc1_scores.min())
#     merged_df =merged_df[["cluster number","Population","TotlowBmi","Totclust",
#         "proportion_of_low_bmi",
#         "VI_climate_Rank_final",
#         "VI_disease_Rank_final",
#         "VI_conflict_Rank_final",
#         "VI_socio_Rank_final","VI_PCA"
#     ]]
#     pca_output = os.path.join(pca_path, f"PCA_{year}_{country}.xlsx")
#     merged_df.to_excel(pca_output, index=False)
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

def pca(output_folder, year, country):
    vi_path = os.path.join(output_folder, str(year), "VI")
    pca_path = os.path.join(output_folder, str(year), "PCA")
    os.makedirs(pca_path, exist_ok=True)

    expected_choices = ["climate", "conflict", "disease", "socio"]

    vi_files = {}
    for fname in os.listdir(vi_path):
        fl = fname.lower()
        for choice in expected_choices:
            if choice in fl and fl.endswith(".xlsx") and country.lower() in fl:
                vi_files[choice] = os.path.join(vi_path, fname)

    missing = [ch for ch in expected_choices if ch not in vi_files]
    if missing:
        print(f"⚠️ Missing VI files for: {missing}. PCA will use only available data.")

    # load available dataframes
    dfs = {}
    for choice, path in vi_files.items():
        try:
            dfs[choice] = pd.read_excel(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # merge on cluster number (outer)
    merged_df = None
    for choice, df in dfs.items():
        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = merged_df.merge(df, on="cluster number", how="outer", suffixes=("", f"_{choice}"))

    merged_df["proportion_of_low_bmi"] = merged_df["TotlowBmi"].div(merged_df["Totclust"]).replace([np.inf, -np.inf], np.nan)
    vi_cols = ["VI_climate_Rank_final", "VI_disease_Rank_final",
           "VI_conflict_Rank_final", "VI_socio_Rank_final"]
    final_df = merged_df[vi_cols].copy()
    final_df = final_df.apply(lambda s: s.fillna(s.median()))
    merged_df[vi_cols] = final_df
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(final_df) 
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(X_scaled).ravel()  # shape (n,)
    merged_df["VI_PCA_raw"] = pc1_scores
    merged_df["VI_PCA"] = (pc1_scores - pc1_scores.min()) / (pc1_scores.max() - pc1_scores.min())
    merged_df =merged_df[["cluster number","Population","TotlowBmi","Totclust",
        "proportion_of_low_bmi",
        "VI_climate_Rank_final",
        "VI_disease_Rank_final",
        "VI_conflict_Rank_final",
        "VI_socio_Rank_final","VI_PCA"
    ]]
    pca_output = os.path.join(pca_path, f"PCA_{year}_{country}.xlsx")
    merged_df.to_excel(pca_output, index=False)

