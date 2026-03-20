import os
from typing import Optional

import pandas as pd


def load_raw_data(
    data_path: Optional[str] = None,
    nrows: Optional[int] = None,
    usecols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load the raw dataset from CSV.

    Parameters
    ----------
    data_path : str, optional
        Path to the dataset CSV. Defaults to './dataset.csv' in project root.
    nrows : int, optional
        Number of rows to read (useful for quick experiments).
    usecols : list, optional
        Optional list of columns to read.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with identifier columns dropped.
    """
    if data_path is None:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.csv")

    df = pd.read_csv(data_path, nrows=nrows, usecols=usecols)

    # Drop high-cardinality identifier columns
    cols_to_drop = [col for col in ["nameOrig", "nameDest"] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


__all__ = ["load_raw_data"]

