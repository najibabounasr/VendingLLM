import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import Parallel, delayed
import optuna
import warnings
from best_params import xgboost_params, lightgbm_params
import logging
    
# Function to optimize parameters using Optuna
def optimize_params(model_name, X_train_scaled, y_train, X_test_scaled, y_test,n_trials):
    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e1),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e1)
            }
            model = XGBRegressor(**params, verbosity=0)
        elif model_name == 'LightGBM':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1e1),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 1e1)
            }
            model = LGBMRegressor(**params, verbosity=-1)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials,n_jobs=1)
    return study.best_params


# Function to compute MSE scores
def compute_mse_scores(X_train, X_test, y_train, y_test, features):
    X_train = X_train[features].dropna()
    X_test = X_test[features].dropna()
    y_train = y_train.dropna()
    y_test = y_test.dropna()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mse_scores = {'XGBoost': [], 'LightGBM': []}
    best_params = {}

    # def objective(trial, model_name):
    #     # params = optimize_params(trial, model_name)
    #     params = optimize_params(model_name, X_train_scaled, y_train, X_test_scaled, y_test,n_trials=5)
    #     if model_name == 'XGBoost':
    #         model = XGBRegressor(**params, verbosity=0)
    #     elif model_name == 'LightGBM':
    #         model = LGBMRegressor(**params, verbose=-1)
    #     model.fit(X_train_scaled, y_train)
    #     y_pred = model.predict(X_test_scaled)
    #     mse = mean_squared_error(y_test, y_pred)
    #     mse_scores[model_name].append(mse)
    #     return mse

    # for model_name in ['XGBoost', 'LightGBM']:
    #     study = optuna.create_study(direction='minimize')
    #     study.optimize(lambda trial: objective(trial, model_name), n_trials=50)
    #     best_params[model_name] = study.best_params

    aggregated_mse = sum([np.mean(mse_scores[model]) for model in mse_scores])
    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}

    return mse_scores, aggregated_mse, best_params

# Function to compute baseline MSE scores without adding any new feature
def compute_baseline_mse(X_train, X_test, y_train, y_test, base_features):
    X_train = X_train[base_features].dropna().values
    X_test = X_test[base_features].dropna().values
    y_train = y_train.dropna().values.ravel()
    y_test = y_test.dropna().values.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mse_scores = {'XGBoost': [], 'LightGBM': []}

    models = {
        'XGBoost': XGBRegressor(**xgboost_params, verbosity=0),
        'LightGBM': LGBMRegressor(**lightgbm_params, verbosity=-1)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores[model_name].append(mean_squared_error(y_test, y_pred))

    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}
    aggregated_mse = sum(mse_scores.values())

    return mse_scores, aggregated_mse

# Function to compute MSE scores after adding a feature
def compute_mse_with_added_feature(X_train, X_test, y_train, y_test, base_features, add_feature):
    X_train = X_train[base_features + [add_feature]].dropna().values
    X_test = X_test[base_features + [add_feature]].dropna().values
    y_train = y_train.dropna().values.ravel()
    y_test = y_test.dropna().values.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mse_scores = {'XGBoost': [], 'LightGBM': []}

    models = {
        'XGBoost': XGBRegressor(**xgboost_params, verbosity=0),
        'LightGBM': LGBMRegressor(**lightgbm_params, verbosity=-1)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores[model_name].append(mean_squared_error(y_test, y_pred))

    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}
    aggregated_mse = sum(mse_scores.values())

    return mse_scores, aggregated_mse

# Automated Feature Extraction using TSFRESH
def extract_tsfresh_features(data, column_id, column_sort, default_fc_parameters):
    df_long = roll_time_series(data, column_id=column_id, column_sort=column_sort)
    extracted_features = extract_features(df_long, column_id=column_id, column_sort=column_sort, default_fc_parameters=default_fc_parameters)
    extracted_features = extracted_features.dropna(axis=1, how='any')  # Drop columns with NaNs
    return extracted_features

# Function to evaluate a single feature
def evaluate_feature(feature, tsfresh_features_train, tsfresh_features_test, X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, base_features, aggregated_baseline_mse, all_added_features):
    if feature in all_added_features:
        return None
    if feature not in tsfresh_features_train.columns or feature not in tsfresh_features_test.columns:
        return None
    temp_X_train = pd.concat([X_train_transformed, tsfresh_features_train[[feature]]], axis=1)
    temp_X_test = pd.concat([X_test_transformed, tsfresh_features_test[[feature]]], axis=1)
    mse_scores, aggregated_mse = compute_mse_with_added_feature(temp_X_train, temp_X_test, y_train_transformed, y_test_transformed, base_features, feature)
    improvement = aggregated_baseline_mse - aggregated_mse
    improvement_status = "improved" if improvement > 0 else "worsened"
    return (feature, aggregated_mse, improvement, improvement_status, mse_scores)


# Function to compute MSE scores after dropping a feature
def compute_mse_with_dropped_feature(X_train, X_test, y_train, y_test, base_features, drop_feature):
    X_train = X_train[[f for f in base_features if f != drop_feature]].dropna().values
    X_test = X_test[[f for f in base_features if f != drop_feature]].dropna().values
    y_train = y_train.dropna().values.ravel()
    y_test = y_test.dropna().values.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mse_scores = {'XGBoost': [], 'LightGBM': []}

    models = {
        'XGBoost': XGBRegressor(**xgboost_params, verbosity=0),
        'LightGBM': LGBMRegressor(**lightgbm_params, verbosity=-1)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse_scores[model_name].append(mean_squared_error(y_test, y_pred))

    mse_scores = {model: np.mean(scores) for model, scores in mse_scores.items()}
    aggregated_mse = sum(mse_scores.values())

    return mse_scores, aggregated_mse

# Function to drop features
# def drop_features(train_combined, target, base_features,aggregated_baseline_mse,threshold):
#     aggregated_mse_scores_dropped = []
#     for feature in base_features:
#         mse_scores, aggregated_mse = compute_mse_with_dropped_feature(X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed, base_features, feature)
#         improvement = aggregated_baseline_mse - aggregated_mse
#         improvement_status = "improved" if improvement > threshold else "worsened"
#         aggregated_mse_scores_dropped.append((feature, aggregated_mse, improvement, improvement_status, mse_scores))


#     # Sort and drop the least impactful features if they result in improvement
#     aggregated_mse_scores_dropped.sort(key=lambda x: x[1])
#     features_to_drop = [f for f in aggregated_mse_scores_dropped if f[2] > threshold]

#     if not features_to_drop:
#         print("No features were dropped as they did not improve the model.")
#     else:
#         for feature, _, improvement, _, _ in features_to_drop:
#             base_features.remove(feature)
#             print(f"Feature dropped: {feature}, Improvement: {improvement}")

#     print("Feature Dropping Completed.")
