
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



# def calculate_covrank(df,cols_to_use):
#     cols_to_use += [col for col in ["TotlowBmi", "Totclust"] if col in df.columns]
#     df2 = df[cols_to_use].copy()
#     all_Zeros_dataframe = df2.loc[:, (df2 == 0).all(axis=0)]
#     df2 = df2.loc[:, (df2 != 0).any(axis=0)]
#     df2 = df2[df2['Totclust'] != 0]
#     df2['Unvaccinated'] = df2['Totclust'] - df2['TotlowBmi']
#     response = df2[['TotlowBmi', 'Unvaccinated']]
#     covariates = [col for col in df2.columns if col not in ['Totclust', 'TotlowBmi', 'Unvaccinated']]
#     results = {'model': [], 'AIC': [], 'r2': [], 'pr2': [], 'deviance': [], 'dev_red': []}
#     n_iter = 5
#     propsub = 0.8

#     null_model = sm.GLM(response, sm.add_constant(df2[['Totclust']]), family=sm.families.Binomial())
#     null_result = null_model.fit()
#     null_deviance = getattr(null_result, "deviance", np.nan)

#     for cov in covariates:
#         aics, r2s, pr2s, devs = [], [], [], []
#         for _ in range(n_iter):
#             train_data, test_data = train_test_split(df2, test_size=1 - propsub, random_state=None)
            
#             train_model = sm.GLM(train_data[['TotlowBmi', 'Unvaccinated']], sm.add_constant(train_data[[cov]]), family=sm.families.Binomial())
#             full_model = sm.GLM(response, sm.add_constant(df2[[cov]]), family=sm.families.Binomial())

#             train_result = train_model.fit()
#             full_result = full_model.fit()

#             pred_train = train_result.predict(sm.add_constant(train_data[[cov]]))
#             actual_train = train_data['TotlowBmi'] / train_data['Totclust']
#             pr2 = np.corrcoef(actual_train, pred_train)[0, 1] ** 2 if len(actual_train) > 1 else 0
#             pr2s.append(pr2)

#             aics.append(getattr(full_result, "aic", np.nan))
#             devs.append(getattr(full_result, "deviance", np.nan))
#             r2 = np.corrcoef(df2['TotlowBmi'] / df2['Totclust'], full_result.fittedvalues)[0, 1] ** 2 if df2.shape[0] > 1 else 0
#             r2s.append(r2)

#         results['model'].append(cov)
#         results['AIC'].append(np.nanmean(aics))
#         results['r2'].append(np.nanmean(r2s))
#         results['pr2'].append(np.nanmean(pr2s))
#         results['deviance'].append(np.nanmean(devs))
#         results['dev_red'].append(np.nanmean(devs) / null_deviance if not np.isnan(null_deviance) else np.nan)

#     results_df = pd.DataFrame(results)
#     results_df = results_df[results_df['model'] != 'Unvaccinated']
#     results_df = results_df.sort_values(by='pr2', ascending=False).reset_index(drop=True)

#     results_df['rank1'] = (results_df['pr2'].rank(ascending=True, method='min')).fillna(0).astype(int)
#     try:
#         results_df['rank'] = pd.qcut(results_df['pr2'].fillna(0), q=4, labels=False, duplicates='drop') + 1
#         results_df['rank'] = results_df['rank'].fillna(0).astype(int)
#     except Exception:
#         results_df['rank'] = 0

#     numeric_results = results_df.copy()

#     if not all_Zeros_dataframe.empty:
#         zero_cols = all_Zeros_dataframe.columns.tolist()
#         # print(f"These columns are all NAN or Zeros {zero_cols}")
#         zero_rows = pd.DataFrame([{
#             'model': col,
#             'AIC': 0,
#             'r2': 0,
#             'pr2': 0,
#             'deviance': 0,
#             'dev_red': 0,
#             'rank1': 0,
#             'rank': 0
#         } for col in zero_cols])
#         numeric_results = pd.concat([numeric_results, zero_rows], ignore_index=True)
    

#     return numeric_results
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
from tqdm import tqdm


def calculate_covrank(df, cols_to_use, n_iter_start=5, max_iter=100, tol=0.01, propsub=0.8, plot_convergence=True):
    cols_to_use += [col for col in ["TotlowBmi", "Totclust"] if col in df.columns]
    df = df[cols_to_use].copy()
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df[df['Totclust'] != 0]
    df['Unvaccinated'] = df['Totclust'] - df['TotlowBmi']
    df['prop'] = df['TotlowBmi'] / df['Totclust']

    covariates = [col for col in df.columns if col not in ['Unvaccinated', 'Totclust', 'TotlowBmi', 'prop']]

    # Containers
    mean_pr2s = []
    results_list = []

    n_iter = n_iter_start
    prev_mean_pr2 = None

    # Main iteration loop with tqdm
    for n_iter in tqdm(range(n_iter_start, max_iter + 1), desc="Main Iterations"):
        results = {'model': [], 'AIC': [], 'r2': [], 'pr2': [], 'deviance': [], 'dev_red': []}

        # Random subsample iterations with tqdm
        for _ in tqdm(range(n_iter), desc=f"Sub-Iterations (n_iter={n_iter})", leave=False):
            # Random subsample
            train_data, test_data = train_test_split(df, test_size=1 - propsub)

            # Null model (for deviance reduction)
            null_model = sm.GLM(train_data['prop'],
                                np.ones((len(train_data), 1)),
                                family=sm.families.Binomial(),
                                freq_weights=train_data['Totclust'])
            null_result = null_model.fit()
            null_deviance = null_result.deviance

            # Model loop with tqdm
            for cov in tqdm(covariates, desc="Covariates", leave=False):
                train_model = sm.GLM(train_data['prop'],
                                     sm.add_constant(train_data[[cov]]),
                                     family=sm.families.Binomial(),
                                     freq_weights=train_data['Totclust'])
                train_result = train_model.fit()

                pr2 = np.corrcoef(train_data['prop'], train_result.fittedvalues)[0, 1] ** 2

                full_model = sm.GLM(df['prop'],
                                    sm.add_constant(df[[cov]]),
                                    family=sm.families.Binomial(),
                                    freq_weights=df['Totclust'])
                full_result = full_model.fit()

                r2 = np.corrcoef(df['prop'], full_result.fittedvalues)[0, 1] ** 2

                results['model'].append(cov)
                results['AIC'].append(full_result.aic)
                results['r2'].append(r2)
                results['pr2'].append(pr2)
                results['deviance'].append(full_result.deviance)
                results['dev_red'].append(1 - full_result.deviance / null_deviance)

        results_df = pd.DataFrame(results)
        summary_df = results_df.groupby('model', as_index=False).agg({
            'AIC': 'first',
            'r2': 'first',
            'pr2': 'mean',
            'deviance': 'first',
            'dev_red': 'first'
        }).sort_values(by='pr2', ascending=False).reset_index(drop=True)

        mean_pr2 = summary_df['pr2'].mean()
        mean_pr2s.append(mean_pr2)
        results_list.append(summary_df)

    # Convert results to DataFrame and rank by predictive R2 (pr2)
    results_df = pd.DataFrame(summary_df)
    
    results_df['rank'] = results_df['pr2'].rank(ascending=False)
    results_df = results_df.sort_values(by='pr2', ascending=False).reset_index(drop=True)
    results = results_df.copy()
    results['rank'] = (results['pr2'].rank(ascending=True)).astype(int)
    results['rank1'] = pd.qcut(results['pr2'], q=4, labels=False, duplicates='drop') + 1
    
    return results
