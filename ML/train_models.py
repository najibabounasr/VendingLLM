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
import ML.process_data as prcd
import ML.split_data as split
def _temporal_val_split(X, y, val_frac=0.2):
    n = len(X)
    cut = max(1, int(n * (1 - val_frac)))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]



def optimize_xgboost(X_train, y_train, preprocessor, n_trials: int = 30):
    random.seed(42); np.random.seed(42)
    best_mse, best_params = float('inf'), None

    X_tr, X_val, y_tr, y_val = _temporal_val_split(X_train, y_train, val_frac=0.2)

    for _ in range(n_trials):
        param = {
            'max_depth': random.randint(3, 10),
            'learning_rate': 10 ** random.uniform(-3, -1),
            'n_estimators': random.randint(50, 300),
            'subsample': random.uniform(0.5, 1.0),
            'colsample_bytree': random.uniform(0.5, 1.0),
            'reg_lambda': 10 ** random.uniform(-3, 1),
            'reg_alpha':  10 ** random.uniform(-3, 1),
        }
        model = xgb.XGBRegressor(
            **param, objective='reg:squarederror', n_jobs=-1, verbosity=0
        )
        pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
        pipe.fit(X_tr, y_tr)
        mse = mean_squared_error(y_val, pipe.predict(X_val))
        if mse < best_mse:
            best_mse, best_params = mse, param
    return best_params



def optimize_lightgbm(X_train, y_train, preprocessor, n_trials: int = 30):
    random.seed(42); np.random.seed(42)
    best_mse, best_params = float('inf'), None

    X_tr, X_val, y_tr, y_val = _temporal_val_split(X_train, y_train, val_frac=0.2)

    for _ in range(n_trials):
        param = {
            'num_leaves':        random.randint(31, 255),
            'learning_rate':     10 ** random.uniform(-3, -1),
            'n_estimators':      random.randint(50, 300),
            'bagging_fraction':  random.uniform(0.5, 1.0),
            'feature_fraction':  random.uniform(0.5, 1.0),
            'boosting_type':     random.choice(['gbdt', 'dart']),
            'min_child_samples': random.randint(10, 100),
        }
        model = lgb.LGBMRegressor(
            **param, objective='regression', n_jobs=-1, verbose=-1
        )
        pipe = Pipeline([('preprocess', preprocessor), ('model', model)])
        pipe.fit(X_tr, y_tr)
        mse = mean_squared_error(y_val, pipe.predict(X_val))
        if mse < best_mse:
            best_mse, best_params = mse, param
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
    file_path = "vending_machine_sales.csv"

    # 1) Load CSV, build daily features/target (expects a DF, not a path)
    raw = pd.read_csv(file_path)
    daily = prcd.daily_sales(raw)  # returns columns incl. TransDate, DailyUnits, etc.
    daily = daily.sort_values("TransDate").reset_index(drop=True)

    # 2) Prepare features/target (time-ordered split done inside)
    TEST_SIZE = 0.2
    X_train, X_test, y_train, y_test, preprocessor = split.prepare_features_and_target_v1(
        df=daily,
        target_col="DailyUnits",
        date_col="TransDate",
        test_size=TEST_SIZE,
        add_calendar=False,          # daily already has calendar features; set True if you want more
        drop_cols_leak=None,
        one_hot_cats=True,
        require_min_features=True
    )

    # Keep TransDate for the test slice (since prep drops it from X)
    n = len(daily)
    cut = int(round(n * (1 - TEST_SIZE)))
    test_dates = daily["TransDate"].iloc[cut:].reset_index(drop=True)

    # 3) Hyperparameter search
    print("Optimizing XGBoost hyperparameters...")
    best_params_xgb = optimize_xgboost(X_train, y_train, preprocessor, n_trials=30)
    print("Best XGBoost parameters:", best_params_xgb)

    print("Optimizing LightGBM hyperparameters...")
    best_params_lgb = optimize_lightgbm(X_train, y_train, preprocessor, n_trials=30)
    print("Best LightGBM parameters:", best_params_lgb)

    # 4) Train & evaluate
    xgb_model, xgb_mse, xgb_pred = train_and_evaluate_model(
        model_name="xgboost",
        params=best_params_xgb,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        preprocessor=preprocessor
    )
    print(f"XGBoost MSE: {xgb_mse:.4f}")

    lgb_model, lgb_mse, lgb_pred = train_and_evaluate_model(
        model_name="lightgbm",
        params=best_params_lgb,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        preprocessor=preprocessor
    )
    print(f"LightGBM MSE: {lgb_mse:.4f}")

    # 5) Save summaries
    pd.DataFrame([
        {"model": "XGBoost", "mse": xgb_mse, "params": best_params_xgb},
        {"model": "LightGBM", "mse": lgb_mse, "params": best_params_lgb},
    ]).to_csv("model_performance.csv", index=False)

    predictions_df = pd.DataFrame({
        "TransDate": test_dates,     # explicit date column
        "actual": y_test.reset_index(drop=True),
        "xgboost_pred": pd.Series(xgb_pred),
        "lightgbm_pred": pd.Series(lgb_pred),
    })
    predictions_df.to_csv("model_predictions.csv", index=False)

    print("Training complete. Results saved to model_performance.csv and model_predictions.csv")


if __name__ == '__main__':
    main()