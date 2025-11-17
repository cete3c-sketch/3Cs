
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


class FilenameError(Exception):
    pass




def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '', name)  # keep only alphanumeric
    return name

def best_match(col, expected_cols, threshold=95):
    col_norm = normalize(col)
    candidates = [normalize(c) for c in expected_cols]
    matches = process.extractOne(col_norm, candidates, scorer=fuzz.ratio)
    
    if matches and matches[1] >= threshold:  # matches[1] = score
        # return original expected column (not normalized)
        idx = candidates.index(matches[0])
        return expected_cols[idx]
    return None


class FilenameError(Exception):
    pass


def parse_filename(filename: str):
    """
    Returns (year, country, choice) or raises FilenameError if not all found.
    """
    countries = {"eth", "camroon", "drc", "bf"}
    choices = {"climate", "socio", "disease", "conflict"}

    # strip extension and lowercase
    base = os.path.splitext(filename)[0].lower()

    # split on underscores, dashes or spaces and remove empty tokens
    tokens = [t for t in re.split(r'[_\-\s]+', base) if t]

    # choice should be the last token
    choice = tokens[-1] if tokens and tokens[-1] in choices else None

    # year is the first 4-digit token
    year = next((t for t in tokens if re.fullmatch(r'\d{4}', t)), None)

    # country is any token that matches the allowed country list
    country = next((t for t in tokens if t in countries), None)

    if not (year and country and choice):
        raise FilenameError(f"No correct name format found in filename. "
                            f"parsed -> year={year}, country={country}, choice={choice}")

    return year, country, choice



