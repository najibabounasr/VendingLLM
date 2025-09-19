import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import random
import xgboost as xgb
import lightgbm as lgb


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load the vending machine sales dataset and perform basic cleaning.

    - Removes duplicate rows.
    - Fills missing categorical values with 'Unknown'.
    - Fills missing numeric values with the median of the column.
    - Converts date columns to datetime. The dates are not used directly
      in the model to avoid implicit feature engineering but are kept in the
      returned DataFrame for reference if needed later.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing sales data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = pd.read_csv(file_path)

    # Drop duplicate rows if any
    df = df.drop_duplicates().reset_index(drop=True)

    # Identify categorical and numeric columns
    categorical_cols = [
        'Status', 'Device ID', 'Location', 'Machine', 'Product', 'Category', 'Type'
    ]
    numeric_cols = [
        'Transaction', 'RCoil', 'RPrice', 'RQty', 'MCoil', 'MPrice', 'MQty', 'LineTotal', 'TransTotal'
    ]

    # Fill missing categorical values with 'Unknown'
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Fill missing numeric values with median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Convert date columns to datetime if they exist
    date_cols = ['TransDate', 'Prcd Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


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

def prepare_features_and_target_TS(df: pd.DataFrame, target_col: str = 'LineTotal'):
    """
    Separate the DataFrame into features (X) and target (y). Perform a
    chronological train-test split (time series style) and construct a
    preprocessing pipeline that one-hot encodes categorical features
    and leaves numeric features unchanged.
    """
    # Sort by transaction date to preserve time order
    df = df.sort_values("TransDate").reset_index(drop=True)

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

    # Sequential split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, preprocessor



def optimize_xgboost(X_train, y_train, preprocessor, n_trials: int = 30):
    """
    Perform simple random search hyperparameter optimization for an XGBoost
    regressor. Since external packages like Optuna are unavailable in this
    environment, this function randomly samples parameter combinations
    and selects the one with the lowest training MSE.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features before preprocessing.
    y_train : pd.Series
        Training target.
    preprocessor : ColumnTransformer
        Preprocessing transformer for encoding categorical variables.
    n_trials : int
        Number of random parameter combinations to try.

    Returns
    -------
    dict
        The best set of hyperparameters found.
    """
    best_mse = float('inf')
    best_params = None

    for _ in range(n_trials):
        # Randomly sample hyperparameters within reasonable ranges
        param = {
            'max_depth': random.randint(3, 10),
            'learning_rate': 10 ** random.uniform(-3, -1),  # between 0.001 and 0.1
            'n_estimators': random.randint(50, 300),
            'subsample': random.uniform(0.5, 1.0),
            'colsample_bytree': random.uniform(0.5, 1.0),
            'reg_lambda': 10 ** random.uniform(-3, 1),  # between 0.001 and 10
            'reg_alpha': 10 ** random.uniform(-3, 1),  # between 0.001 and 10
        }
        model = xgb.XGBRegressor(
            **param,
            objective='reg:squarederror',
            n_jobs=-1,
            verbosity=0,
        )
        pipeline = Pipeline(
            steps=[('preprocess', preprocessor), ('model', model)]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_train)
        mse = mean_squared_error(y_train, preds)
        if mse < best_mse:
            best_mse = mse
            best_params = param
    return best_params


def optimize_lightgbm(X_train, y_train, preprocessor, n_trials: int = 30):
    """
    Perform simple random search hyperparameter optimization for a LightGBM
    regressor. External optimization libraries are not available, so this
    function samples random parameter combinations and selects the one
    yielding the lowest training MSE.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features before preprocessing.
    y_train : pd.Series
        Training target.
    preprocessor : ColumnTransformer
        Preprocessing transformer for encoding categorical variables.
    n_trials : int
        Number of random parameter combinations to try.

    Returns
    -------
    dict
        The best set of hyperparameters found.
    """
    best_mse = float('inf')
    best_params = None

    for _ in range(n_trials):
        param = {
            'num_leaves': random.randint(31, 255),
            'learning_rate': 10 ** random.uniform(-3, -1),  # between 0.001 and 0.1
            'n_estimators': random.randint(50, 300),
            'bagging_fraction': random.uniform(0.5, 1.0),
            'feature_fraction': random.uniform(0.5, 1.0),
            'boosting_type': random.choice(['gbdt', 'dart']),
            'min_child_samples': random.randint(10, 100),
        }
        model = lgb.LGBMRegressor(
            **param,
            objective='regression',
            n_jobs=-1,
            verbose=-1,
        )
        pipeline = Pipeline(
            steps=[('preprocess', preprocessor), ('model', model)]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_train)
        mse = mean_squared_error(y_train, preds)
        if mse < best_mse:
            best_mse = mse
            best_params = param
    return best_params


def train_and_evaluate_model(
    model_name: str,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer
):
    """
    Train either an XGBoost or LightGBM model with the provided hyperparameters
    and evaluate it on the test set.

    Parameters
    ----------
    model_name : str
        Name of the model ('xgboost' or 'lightgbm').
    params : dict
        Hyperparameters for the model.
    X_train : pd.DataFrame
        Training features before preprocessing.
    X_test : pd.DataFrame
        Testing features before preprocessing.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Testing target.
    preprocessor : ColumnTransformer
        Preprocessing transformer.

    Returns
    -------
    tuple
        (model_pipeline, test_mse, y_pred), where model_pipeline is the
        trained pipeline, test_mse is the MSE on the test set, and y_pred
        is a numpy array of predictions.
    """
    if model_name == 'xgboost':
        model = xgb.XGBRegressor(
            **params,
            objective='reg:squarederror',
            n_jobs=-1,
            verbosity=0
        )
    elif model_name == 'lightgbm':
        model = lgb.LGBMRegressor(
            **params,
            objective='regression',
            n_jobs=-1,
            verbose=-1
        )
    else:
        raise ValueError("model_name must be 'xgboost' or 'lightgbm'")

    pipeline = Pipeline(
        steps=[('preprocess', preprocessor), ('model', model)]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return pipeline, mse, y_pred


def main():
    # Filepath to the uploaded dataset
    file_path = 'vending_machine_sales.csv'

    # Load and clean data
    df = load_and_clean_data(file_path)

    # Prepare features and target
    X_train, X_test, y_train, y_test, preprocessor = prepare_features_and_target_TS(df, target_col='LineTotal')

    # Optimize hyperparameters for XGBoost
    print("Optimizing XGBoost hyperparameters...")
    best_params_xgb = optimize_xgboost(X_train, y_train, preprocessor, n_trials=30)
    print("Best XGBoost parameters:", best_params_xgb)

    # Optimize hyperparameters for LightGBM
    print("Optimizing LightGBM hyperparameters...")
    best_params_lgb = optimize_lightgbm(X_train, y_train, preprocessor, n_trials=30)
    print("Best LightGBM parameters:", best_params_lgb)

    # Train and evaluate XGBoost
    xgb_model, xgb_mse, xgb_pred = train_and_evaluate_model(
        model_name='xgboost',
        params=best_params_xgb,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor
    )
    print(f"XGBoost MSE: {xgb_mse:.4f}")

    # Train and evaluate LightGBM
    lgb_model, lgb_mse, lgb_pred = train_and_evaluate_model(
        model_name='lightgbm',
        params=best_params_lgb,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor
    )
    print(f"LightGBM MSE: {lgb_mse:.4f}")

    # Save model performance summary
    performance_df = pd.DataFrame([
        {
            'model': 'XGBoost',
            'mse': xgb_mse,
            'params': best_params_xgb
        },
        {
            'model': 'LightGBM',
            'mse': lgb_mse,
            'params': best_params_lgb
        }
    ])
    performance_df.to_csv('model_performance.csv', index=False)

    # Save predictions along with actual values and identifiers
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'xgboost_pred': xgb_pred,
        'lightgbm_pred': lgb_pred,
    })
    # Optionally include identifier columns from the original test set for reference
    id_cols = ['Device ID', 'Product', 'Category', 'TransDate']
    for col in id_cols:
        if col in X_test.columns:
            predictions_df[col] = X_test[col].values
    predictions_df.to_csv('model_predictions.csv', index=False)

    print("Training complete. Results saved to model_performance.csv and model_predictions.csv")


if __name__ == '__main__':
    main()