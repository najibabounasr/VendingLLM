import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import scipy
from typing import List, Optional
import os

#  this should work with all dataframes. 
def prepare_features_and_target_v1(
    df: pd.DataFrame,
    *,
    target_col: str,           # REQUIRED
    date_col: str,             # REQUIRED
    test_size: float = 0.2,
    add_calendar: bool = True, # if False, no calendar cols are added
    drop_cols_leak: Optional[List[str]] = None,
    one_hot_cats: bool = True,
    require_min_features: bool = True  # raise if X has 0 columns
):
    # --- validate
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not in df.columns")
    if date_col not in df.columns:
        raise KeyError(f"date_col '{date_col}' not in df.columns")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be in (0,1)")

    # --- copy & enforce datetime
    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
    dfx = dfx.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # --- drop known leaky columns (e.g., centered rolling)
    if drop_cols_leak:
        cols_to_drop = [c for c in drop_cols_leak if c in dfx.columns]
        if cols_to_drop:
            dfx = dfx.drop(columns=cols_to_drop)

    # --- optional calendar features
    if add_calendar:
        dfx["DayOfWeek"] = dfx[date_col].dt.dayofweek
        dfx["Month"]     = dfx[date_col].dt.month
        dfx["Quarter"]   = dfx[date_col].dt.quarter
        dfx["Day"]       = dfx[date_col].dt.day
        dfx["IsWeekend"] = (dfx["DayOfWeek"] >= 5).astype(int)

    # --- build X/y (drop raw datetime)
    y = dfx[target_col]
    X = dfx.drop(columns=[target_col, date_col], errors="ignore")

    # If the caller turned off calendar features and there are no other columns, handle it
    if X.shape[1] == 0:
        if require_min_features:
            raise ValueError(
                "No feature columns remain after dropping target/date. "
                "Provide features in df or set add_calendar=True."
            )
        # else: leave X empty and let the caller decide (most models will reject)

    # --- sequential split
    n = len(dfx)
    split_idx = int(round(n * (1 - test_size)))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- preprocessor
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    transformers = []
    if one_hot_cats and cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    if not transformers:
        # No usable columns at all; keep remainder='drop' to fail clearly at fit
        preprocessor = ColumnTransformer(transformers=[], remainder="drop")
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")


    # remove ALL NAN:

    num = X_train.select_dtypes(np.number).columns
    X_train[num] = X_train[num].interpolate(method="spline", order=3, limit_direction="both")
    X_test[num]  = X_test[num].interpolate(method="spline", order=3, limit_direction="both")
    y_train = y_train.interpolate(method="spline", order=3, limit_direction="both")
    y_test  = y_test.interpolate(method="spline", order=3, limit_direction="both")

    # ensure a per-target output dir
    outdir = os.path.join("data", target_col)
    os.makedirs(outdir, exist_ok=True)

    # save with a consistent date index name
    X_train.to_csv(os.path.join(outdir, "X_train.csv"), index=True, index_label=date_col)
    X_test.to_csv(os.path.join(outdir, "X_test.csv"), index=True, index_label=date_col)

    # y as 1-col CSVs; preserve same index label
    y_train.to_frame(name=target_col).to_csv(os.path.join(outdir, "y_train.csv"), index=True, index_label=date_col)
    y_test.to_frame(name=target_col).to_csv(os.path.join(outdir, "y_test.csv"), index=True, index_label=date_col)


    return X_train, X_test, y_train, y_test, preprocessor









def prepare_features_and_target_NO_TS(df: pd.DataFrame, target_col: str = 'LineTotal'):
    """
    Separate the DataFrame into features (X) and target (y). Perform a
    train-test split and construct a preprocessing pipeline that one-hot
    encodes categorical features and leaves numeric features unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame after cleaning.
    target_col : str, optional
        Name of the target column to predict, by default 'LineTotal'.

    Returns
    -------
    tuple
        A tuple (X_train, X_test, y_train, y_test, preprocessor) where
        preprocessor is the fitted ColumnTransformer.
    """
    # Separate target and features
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Create preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor

def prepare_features_and_target_TS(df: pd.DataFrame, target_col: str = "TotalSales"):
    df = df.sort_values("TransDate").reset_index(drop=True)

    # Target and features
    y = df[target_col]
    X = df.drop(columns=[target_col, "TransDate"])  # drop raw date col, keep engineered ones

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # Sequential split: 80/20
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, preprocessor