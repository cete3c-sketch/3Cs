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
# from .column_maching import normalize, best_match, parse_filename

# from .covrank import calculate_covrank

# climate_columns = ['cluster number','Population',
#  'Drought',
#  'WSDI',
#  'Flood Occurrence',
#  'Flood Fatalities',
#  'Flood Severity',
#  'Earthquake Severity','TotlowBmi',
#  'Totclust']
# conflict_names = ['cluster number','Population',
#  'Fatalities',
#  'Conflict Occurrence',
# 'TotlowBmi',
#  'Totclust']
# disease_names = ['cluster number','Population',
#  'CCHF 1',
#  'CCHF 2',
#  'CCHF 3a',
#  'CCHF 3b',
#  'Ebola Virus 1',
#  'Ebola Virus 2',
#  'Ebola Virus 3a',
#  'Ebola Virus 3b',
#  'Lassa Fever 1',
#  'Lassa Fever 2',
#  'Lassa Fever 3a',
#  'Lassa Fever 3b',
#  'Marburg Virus 1',
#  'Marburg Virus 2',
#  'Marburg Virus 3a',
#  'Marburg Virus 3b',
#  'Malaria Prevalence',
#  'TotlowBmi',
#  'Totclust']
# socio_names =[
#     "cluster number",
#     "Population",
#     "Adolescent",
#     "Pregnant Women",
#     "Non-pregnant Women",
#     "Child Age 0 to 5yrs",
#     "Parity >=3",
#     "Age 15 to 19yrs",
#     "Age 20 to 24yrs",
#     "Age 25 to 29yrs",
#     "Age 30 to 39yrs",
#     "Age 40 to 49yrs",
#     "Rural",
#     "Uneducated",
#     "Primary Education",
#     "Unimproved source of drinking water",
#     "Toilet unavailable",
#     "Unimproved toilet",
#     "Large family",
#     "Households with >=2 children under 5yrs",
#     "Male Household Head",
#     "Households with >=2 births last 5yrs",
#     "Poorer WI",
#     "Poorest WI",
#     "Middle WI",
#     "Households with 2 births last year",
#     "Currently pregnant",
#     "Gestational Age 0 to 4months",
#     "Currently breastfeeding",
#     "Partner is uneducated",
#     "Partner has primary education",
#     "Partner is unemployed",
#     "Partner is unskilled laborer",
#     "Partner is farmer",
#     "Unemployed",
#     "Unskilled laborer",
#     "Farmer",
#     "Partner is healthcare Decision-Maker",
#     "Partner is household Decision-Maker",
#     "Partner decides family visits",
#     "Partner is financial Decision-maker",
#     "TotlowBmi",
#     "Totclust"
# ]

# class FilenameError(Exception):
#     pass





# def Initialize_datafrmae_with_covrank(input_path, output_folder):
    
#     # --- start original logic (kept mostly unchanged) ---
#     _, ext = os.path.splitext(input_path)
#     ext = ext.lower()

#     if ext in [".xls", ".xlsx"]:
#         data = pd.read_excel(input_path)
#     elif ext == ".csv":
#         data = pd.read_csv(input_path)
    
#     file_name = input_path.stem
    
#     year,country,choice=parse_filename(file_name)
#     if choice == "climate":
#         expected_cols = climate_columns
#     elif choice == "conflict":
#         expected_cols = conflict_names
#     elif choice == "disease":
#         expected_cols = disease_names
#     elif choice == "socio":
#         expected_cols = socio_names
#     else:
#         raise ValueError("Invalid choice! Please choose from climate, conflict, disease, or socio.")

#     available_cols = []
#     missing_columns = []

#     for expected in expected_cols:
#         match = best_match(expected, data.columns, threshold=95)
#         if match:
#             available_cols.append(match)
#         else:
#             missing_columns.append(expected)

#     if missing_columns:
#         con = input(
#             f"These are missing columns {missing_columns}. "
#             "Do you want to continue? (type 'continue' or 'exit'): "
#         ).strip()

#         # check with regex for "continue" (case-insensitive)
#         if not re.fullmatch(r"continue", con, re.IGNORECASE):
#             # print("Exiting function...")
#             return None, None, None

#     df = data[available_cols].copy()

    

#     exclude_exact = {"cluster number", "TotlowBmi", "Totclust", "Population"}
#     cols_to_use = [col for col in df.columns if col not in exclude_exact]
    
#     for col in cols_to_use:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     df=df.dropna()
#     scaler = MinMaxScaler()

#     # Convert all columns to numeric if possible, coercing non-numeric to NaN
    

#     # Now safely replace NaN with 0 before scaling
#     if cols_to_use:
#         df.loc[:, cols_to_use] = scaler.fit_transform(df[cols_to_use].fillna(0).astype(float))




#     # ensure output folder exists
#     os.makedirs(os.path.join(output_folder, str(year)), exist_ok=True)
#     os.makedirs(os.path.join(output_folder, str(year),str("Data")), exist_ok=True)
#     main_csv_save = os.path.join(output_folder,f"{year}",f"{"Data"}",f"{country}_{year}_{choice}.csv")
#     df.to_csv(main_csv_save, index=False)

#     numeric_results=calculate_covrank(df,cols_to_use)
#     os.makedirs(os.path.join(output_folder, str(year),str("covrank")), exist_ok=True)
#     output_cverank = os.path.join(output_folder,f"{year}",f"covrank", f"covrank_{year}_{country}_{choice}.csv")
#     numeric_results.to_csv(output_cverank, index=False)
#     # --- end original logic ---
 
#     return choice, main_csv_save, output_cverank,year,country

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

from column_maching import normalize,best_match,parse_filename
from covrank import calculate_covrank
climate_columns = ['cluster number','Population',
 'Drought',
 'WSDI',
 'Flood Occurrence',
 'Flood Fatalities',
 'Flood Severity',
 'Earthquake Severity','TotlowBmi',
 'Totclust']
conflict_names = ['cluster number','Population',
 'Fatalities',
 'Conflict Occurrence',
'TotlowBmi',
 'Totclust']
disease_names = ['cluster number','Population',
 'CCHF 1',
 'CCHF 2',
 'CCHF 3a',
 'CCHF 3b',
 'Ebola Virus 1',
 'Ebola Virus 2',
 'Ebola Virus 3a',
 'Ebola Virus 3b',
 'Lassa Fever 1',
 'Lassa Fever 2',
 'Lassa Fever 3a',
 'Lassa Fever 3b',
 'Marburg Virus 1',
 'Marburg Virus 2',
 'Marburg Virus 3a',
 'Marburg Virus 3b',
 'Malaria Prevalence',
 'TotlowBmi',
 'Totclust']
socio_names =[
    "cluster number",
    "Population",
    "Adolescent",
    "Pregnant Women",
    "Non-pregnant Women",
    "Child Age 0 to 5yrs",
    "Parity >=3",
    "Age 20 to 24yrs",
    "Age 25 to 29yrs",
    "Age 30 to 39yrs",
    "Age 40 to 49yrs",
    "Rural",
    "Uneducated",
    "Primary Education",
    "Unimproved source of drinking water",
    "Toilet unavailable",
    "Unimproved toilet",
    "Large family",
    "Households with >=2 children under 5yrs",
    "Male Household Head",
    "Households with >=2 births last 5yrs",
    "Poorer WI",
    "Poorest WI",
    "Middle WI",
    "Households with 2 births last year",
    "Currently pregnant",
    "Gestational Age 0 to 4months",
    "Currently breastfeeding",
    "Partner is uneducated",
    "Partner has primary education",
    "Partner is unemployed",
    "Partner is unskilled laborer",
    "Partner is farmer",
    "Unemployed",
    "Unskilled laborer",
    "Farmer",
    "Partner is healthcare Decision-Maker",
    "Partner is household Decision-Maker",
    "Partner decides family visits",
    "Partner is financial Decision-maker",
    "TotlowBmi",
    "Totclust"
]

class FilenameError(Exception):
    pass





def Initialize_datafrmae_with_covrank(input_path, output_folder):
    
    # --- start original logic (kept mostly unchanged) ---
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()

    if ext in [".xls", ".xlsx"]:
        data = pd.read_excel(input_path)
    elif ext == ".csv":
        data = pd.read_csv(input_path)
    
    file_name = input_path.stem
    
    year,country,choice=parse_filename(file_name)
    if choice == "climate":
        expected_cols = climate_columns
    elif choice == "conflict":
        expected_cols = conflict_names
    elif choice == "disease":
        expected_cols = disease_names
    elif choice == "socio":
        expected_cols = socio_names
    else:
        raise ValueError("Invalid choice! Please choose from climate, conflict, disease, or socio.")

    available_cols = []
    missing_columns = []

    for expected in expected_cols:
        match = best_match(expected, data.columns, threshold=95)
        if match:
            available_cols.append(match)
        else:
            missing_columns.append(expected)

    if missing_columns:
        con = input(
            f"These are missing columns {missing_columns}. "
            "Do you want to continue? (type 'continue' or 'exit'): "
        ).strip()

        # check with regex for "continue" (case-insensitive)
        if not re.fullmatch(r"continue", con, re.IGNORECASE):
            # print("Exiting function...")
            return None, None, None

    df = data[available_cols].copy()

    

    exclude_exact = {"cluster number", "TotlowBmi", "Totclust", "Population"}
    cols_to_use = [col for col in df.columns if col not in exclude_exact]
    
    for col in cols_to_use:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if col in ["Fatalities", "Conflict Occurrence"]:
            df[col] = df[col].fillna(0)

    len_df_1=len(df)

    df=df.dropna()

    len_df_2=len(df)
    rows_drop=len_df_1-len_df_2
    print(f"Rows drop are {rows_drop} of {year} {choice} {country}")

    scaler = MinMaxScaler()

    # Convert all columns to numeric if possible, coercing non-numeric to NaN
    

    # Now safely replace NaN with 0 before scaling
    if cols_to_use:
        df.loc[:, cols_to_use] = scaler.fit_transform(df[cols_to_use])




    # ensure output folder exists
    os.makedirs(os.path.join(output_folder, str(year)), exist_ok=True)
    os.makedirs(os.path.join(output_folder, str(year),str("Data")), exist_ok=True)
    main_csv_save = os.path.join(output_folder,f"{year}",f"{"Data"}",f"{country}_{year}_{choice}.csv")
    df.to_csv(main_csv_save, index=False)

    numeric_results=calculate_covrank(df,cols_to_use)
    os.makedirs(os.path.join(output_folder, str(year),str("covrank")), exist_ok=True)
    output_cverank = os.path.join(output_folder,f"{year}",f"covrank", f"covrank_{year}_{country}_{choice}.csv")
    numeric_results.to_csv(output_cverank, index=False)
    # --- end original logic ---
 
    return choice, main_csv_save, output_cverank,year,country




