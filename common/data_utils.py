"""
Data loading and splitting helpers shared across subtasks.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file into a pandas DataFrame."""
    return pd.read_parquet(path)


def train_val_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/validation split.

    Parameters
    ----------
    df : DataFrame
        Full labeled dataset.
    label_col : str
        Name of the label column.
    test_size : float
        Fraction for validation set.
    random_state : int
        Random seed.

    Returns
    -------
    (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df, val_df
