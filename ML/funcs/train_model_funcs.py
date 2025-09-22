import pandas as pd
import numpy as np
import mlflow
import optuna
from sklearn.metrics import mean_squared_error
# from autogluon.tabular import TabularPredictor
# import catboost as cb
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb

def clean_data(X, y):
    combined = pd.concat([X, y], axis=1)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.fillna(method='ffill', inplace=True)
    combined.fillna(method='bfill', inplace=True)
    return combined.iloc[:, :-1], combined.iloc[:, -1]

def save_best_params(params_dict, filename='best_params.py'):
    with open(filename, 'w') as f:
        f.write("best_params = ")
        f.write(str(params_dict))
        f.write('\n')

def get_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id       

# def objective_autogluon(trial, X_train, y_train, X_test, y_test, target):
#     experiment_id = get_experiment_id("AutoGluon Optimization")
#     mlflow.set_experiment(experiment_id)
    
#     param_grid = {
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
#         'num_boost_round': trial.suggest_int('num_boost_round', 50, 100),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 50),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
#         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-4, 1e+1),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-4, 1e+1),
#     }
    
#     with mlflow.start_run():
#         # Combine X and y for AutoGluon
#         train_data = pd.concat([X_train, y_train], axis=1)
#         train_data.columns = list(X_train.columns) + [target]  # Ensure the target column is named correctly
        
#         predictor = TabularPredictor(label=target, eval_metric='rmse').fit(
#             train_data=train_data,
#             hyperparameters={'GBM': param_grid},
#             num_bag_folds=5,
#             ag_args_fit={'num_gpus': 0, 'num_cpus': 1}  # Ensure at least 1 CPU is allocated
#         )
#         test_predictions = predictor.predict(X_test)
#         train_predictions = predictor.predict(X_train)
        
#         test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
#         return test_rmse


def objective_catboost(trial,X_train,y_train,X_test,y_test):
    experiment_id = get_experiment_id("CatBoost Optimization")
    mlflow.set_experiment(experiment_id)
    
    param_grid = {
        'depth': trial.suggest_categorical('depth', [8, 12]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'iterations': trial.suggest_int('iterations', 50, 200),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 1e+1),
        'border_count': trial.suggest_categorical('border_count', [24, 26, 30]),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 10),
        'random_strength': trial.suggest_uniform('random_strength', 0, 1),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 10, 50)
    }
    with mlflow.start_run():
        model = cb.CatBoostRegressor(**param_grid, verbose=0)
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        return test_rmse

def objective_knn(trial,X_train,y_train,X_test,y_test):
    experiment_id = get_experiment_id("KNN Optimization")
    mlflow.set_experiment(experiment_id)
    
    param_grid = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 50),
        'p': trial.suggest_int('p', 1, 2)
    }
    with mlflow.start_run():
        model = KNeighborsRegressor(**param_grid)
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        return test_rmse

def objective_lgb(trial,X_train,y_train,X_test,y_test):
    experiment_id = get_experiment_id("LightGBM Optimization")
    mlflow.set_experiment(experiment_id)
    
    param_grid = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0)
    }
    with mlflow.start_run():
        model = lgb.LGBMRegressor(**param_grid)
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        return test_rmse

def objective_xgb(trial,X_train,y_train,X_test,y_test):
    experiment_id = get_experiment_id("XGBoost Optimization")
    mlflow.set_experiment(experiment_id)
    
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-4, 1e+1),
        'alpha': trial.suggest_loguniform('alpha', 1e-4, 1e+1)
    }
    with mlflow.start_run():
        model = xgb.XGBRegressor(**param_grid)
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        return test_rmse

def optimize_model(objective_function, model_name):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_function, n_trials=100)
    best_params = study.best_params
    print(f"Best parameters for {model_name}: {best_params}")
    save_best_params(best_params, filename=f'{model_name}_best_params.py')

#  Extra error handling

# def optimize_model(objective_function, model_name):
#     study = optuna.create_study(direction='minimize')
#     try:
#         study.optimize(objective_function, n_trials=100)
#     except Exception as e:
#         print(f"Optimization failed: {e}")
#     print(f"Best parameters: {study.best_params}")
#     print(f"Best value: {study.best_value}")
#     return study