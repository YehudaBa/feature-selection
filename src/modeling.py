import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

import config as cnfg


def benchmark_xgboost(X, y, test_size=0.2, n_estimators=100, early_stopping_rounds=10, cv=5):
    """
    Runs an XGBoost benchmark model for either classification or regression based on the configuration.

    Parameters:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Target variable.
        test_size (float): Proportion of data for the test set.
        n_estimators (int): Number of boosting rounds.
        early_stopping_rounds (int): Stops training if validation score doesn't improve.
        cv (int): Number of cross-validation folds.

    Returns:
        tuple: Results from the appropriate benchmark function (classification or regression).
    """
    if cnfg.model_type == "classification":
        return benchmark_xgboost_classification(X, y, test_size, n_estimators, early_stopping_rounds, cv)
    elif cnfg.model_type == "regression":
        return benchmark_xgboost_regression(X, y, test_size, n_estimators, early_stopping_rounds, cv)


def benchmark_xgboost_regression(X, y, test_size=0.2, n_estimators=100, early_stopping_rounds=10, cv=5,
                                 random_state=42):
    """
    Runs an XGBoost regression benchmark model.

    Parameters:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Target variable.
        test_size (float): Proportion of data for the test set.
        n_estimators (int): Number of boosting rounds.
        early_stopping_rounds (int): Stops training if validation score doesn't improve.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Trained XGBoost model, test RMSE, and cross-validation RMSE.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_state
    )

    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

    y_pred = model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-cv_scores.mean())

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Cross-Validation RMSE: {cv_rmse:.4f}")
    return model, test_rmse, cv_rmse


def benchmark_xgboost_classification(X, y, test_size=0.2, n_estimators=100, early_stopping_rounds=10, cv=5,
                                     random_state=42):
    """
    Runs an XGBoost classification benchmark model.

    Parameters:
        X (pd.DataFrame or np.array): Feature matrix.
        y (pd.Series or np.array): Target variable.
        test_size (float): Proportion of data for the test set.
        n_estimators (int): Number of boosting rounds.
        early_stopping_rounds (int): Stops training if validation score doesn't improve.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Trained XGBoost model, test F1 score, and cross-validation F1 score.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state
    )

    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

    y_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
    cv_f1 = cv_scores.mean()

    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Cross-Validation F1 Score: {cv_f1:.4f}")

    return model, test_f1, cv_f1
