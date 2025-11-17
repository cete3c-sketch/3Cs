
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



def calculate_covrank(df,cols_to_use):
    cols_to_use += [col for col in ["TotlowBmi", "Totclust"] if col in df.columns]
    df2 = df[cols_to_use].copy()
    all_Zeros_dataframe = df2.loc[:, (df2 == 0).all(axis=0)]
    df2 = df2.loc[:, (df2 != 0).any(axis=0)]
    df2 = df2[df2['Totclust'] != 0]
    df2['Unvaccinated'] = df2['Totclust'] - df2['TotlowBmi']
    response = df2[['TotlowBmi', 'Unvaccinated']]
    covariates = [col for col in df2.columns if col not in ['Totclust', 'TotlowBmi', 'Unvaccinated']]
    results = {'model': [], 'AIC': [], 'r2': [], 'pr2': [], 'deviance': [], 'dev_red': []}
    n_iter = 5
    propsub = 0.8

    null_model = sm.GLM(response, sm.add_constant(df2[['Totclust']]), family=sm.families.Binomial())
    null_result = null_model.fit()
    null_deviance = getattr(null_result, "deviance", np.nan)

    for cov in covariates:
        aics, r2s, pr2s, devs = [], [], [], []
        for _ in range(n_iter):
            train_data, test_data = train_test_split(df2, test_size=1 - propsub, random_state=None)
            
            train_model = sm.GLM(train_data[['TotlowBmi', 'Unvaccinated']], sm.add_constant(train_data[[cov]]), family=sm.families.Binomial())
            full_model = sm.GLM(response, sm.add_constant(df2[[cov]]), family=sm.families.Binomial())

            train_result = train_model.fit()
            full_result = full_model.fit()

            pred_train = train_result.predict(sm.add_constant(train_data[[cov]]))
            actual_train = train_data['TotlowBmi'] / train_data['Totclust']
            pr2 = np.corrcoef(actual_train, pred_train)[0, 1] ** 2 if len(actual_train) > 1 else 0
            pr2s.append(pr2)

            aics.append(getattr(full_result, "aic", np.nan))
            devs.append(getattr(full_result, "deviance", np.nan))
            r2 = np.corrcoef(df2['TotlowBmi'] / df2['Totclust'], full_result.fittedvalues)[0, 1] ** 2 if df2.shape[0] > 1 else 0
            r2s.append(r2)

        results['model'].append(cov)
        results['AIC'].append(np.nanmean(aics))
        results['r2'].append(np.nanmean(r2s))
        results['pr2'].append(np.nanmean(pr2s))
        results['deviance'].append(np.nanmean(devs))
        results['dev_red'].append(np.nanmean(devs) / null_deviance if not np.isnan(null_deviance) else np.nan)

    results_df = pd.DataFrame(results)
    results_df = results_df[results_df['model'] != 'Unvaccinated']
    results_df = results_df.sort_values(by='pr2', ascending=False).reset_index(drop=True)

    results_df['rank1'] = (results_df['pr2'].rank(ascending=True, method='min')).fillna(0).astype(int)
    try:
        results_df['rank'] = pd.qcut(results_df['pr2'].fillna(0), q=4, labels=False, duplicates='drop') + 1
        results_df['rank'] = results_df['rank'].fillna(0).astype(int)
    except Exception:
        results_df['rank'] = 0

    numeric_results = results_df.copy()

    if not all_Zeros_dataframe.empty:
        zero_cols = all_Zeros_dataframe.columns.tolist()
        # print(f"These columns are all NAN or Zeros {zero_cols}")
        zero_rows = pd.DataFrame([{
            'model': col,
            'AIC': 0,
            'r2': 0,
            'pr2': 0,
            'deviance': 0,
            'dev_red': 0,
            'rank1': 0,
            'rank': 0
        } for col in zero_cols])
        numeric_results = pd.concat([numeric_results, zero_rows], ignore_index=True)
    
    return numeric_results