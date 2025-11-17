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

import argparse




from .create_VI import create_VI
from .create_dataframe import Initialize_datafrmae_with_covrank
from .pca import pca








def main_function(input_folder, output_folder):
    """
    Process all input files (create covrank + VI). Then run PCA once for each unique (year, country).
    """
    processed_pairs = set()  # stores (year, country) tuples for which we've created VI
    failed_files = []

    for file in os.listdir(input_folder):
        input_path = Path(input_folder) / file
        try:
            choice, main_csv_save, output_cverank, year, country = Initialize_datafrmae_with_covrank(input_path, output_folder)
        except Exception as e:
            print(f"Error processing covrank for file {file}: {e}")
            failed_files.append((file, str(e)))
            continue

        try:
            # create_VI as before (this writes the VI file)
            create_VI(output_folder, year, country, choice, main_csv_save)
            processed_pairs.add((year, country))
        except Exception as e:
            print(f"Error creating VI for {file} (year={year}, country={country}): {e}")
            failed_files.append((file, str(e)))
            continue

    # After processing all files, run PCA once per unique (year, country)
    for year, country in sorted(processed_pairs):
        try:
            print(f"Running PCA for year={year}, country={country} ...")
            pca(output_folder, year, country)
        except Exception as e:
            print(f"Error running PCA for year={year}, country={country}: {e}")
            failed_files.append((f"PCA_{year}_{country}", str(e)))

    if failed_files:
        print("\nSummary: some files failed during processing:")
        for name, err in failed_files:
            print(f" - {name}: {err}")
    else:
        print("\nAll files processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run covariate ranking")
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    args = parser.parse_args()
    main_function(args.input_folder, args.output_folder)


#  python bashfile.py ..\..\camrooon\new\5_flood_camroon\output ..\output_file\camroon
#  python bashfile.py ..\Burkina_faso\new\5_flood_BF\output output_file\burkina
#  python bashfile.py ..\DRC\new\5_flood_drc\output output_file\DRC
#  python bashfile.py ..\Ethopia\6_food_insurity\food_insecurity_output_ethopia output_file\Ethopia




