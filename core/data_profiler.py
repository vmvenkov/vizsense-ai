import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    numeric_cols: list
    categorical_cols: list
    datetime_cols: list
    missing_rate: float
    cardinality: dict  # col -> n_unique
    skewness: dict     # numeric col -> skew

def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    n_rows, n_cols = df.shape

    missing_rate = float(df.isna().mean().mean()) if n_rows and n_cols else 0.0

    datetime_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            datetime_cols.append(c)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if (c not in numeric_cols and c not in datetime_cols)]

    cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}

    skewness = {}
    for c in numeric_cols:
        s = df[c].dropna()
        skewness[c] = float(s.skew()) if len(s) > 2 else 0.0

    return DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        missing_rate=missing_rate,
        cardinality=cardinality,
        skewness=skewness,
    )
