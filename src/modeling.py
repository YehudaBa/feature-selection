from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import config as cnfg
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import xgboost as xgb
import numpy as np

def benchmark_model(X_train, X_test, y_train, y_test, task="regression", model_type="boosting"):
    """
    Runs a benchmark model that evaluates all features.

    Parameters:
    X_train, X_test, y_train, y_test: Train-test split
    task (str): "regression" or "classification"
    model_type (str): "ridge", "lasso", "logistic", or "boosting"

    Returns:
    float: Performance metric (MSE for regression, Accuracy for classification)
    """
    if task not in ["regression", "classification"]:
        raise ValueError("Invalid task type")
    if task == "regression":
        if model_type == "ridge":
            model = Ridge(alpha=0.01)  # Ensures all features are used
        elif model_type == "lasso":
            model = Lasso(alpha=0.001)  # Very low alpha to keep most features
        elif model_type == "boosting":
            model = GradientBoostingRegressor(n_estimators=100)
        else:
            raise ValueError("Invalid model type for regression")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = mean_squared_error(y_test, y_pred)
        print(f"Benchmark Model ({model_type.capitalize()}) MSE: {metric:.4f}")

    elif task == "classification":
        if model_type == "logistic":
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        elif model_type == "boosting":
            model = GradientBoostingClassifier(n_estimators=100)
        else:
            raise ValueError("Invalid model type for classification")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metric = accuracy_score(y_test, y_pred)
        print(f"Benchmark Model ({model_type.capitalize()}) Accuracy: {metric:.4f}")

    return metric



def benchmark_xgboost(X, y, test_size=0.2, n_estimators=100, early_stopping_rounds=10, cv=5):
    if cnfg.model_type == "classification":
        return benchmark_xgboost_classification(X, y, test_size, n_estimators, early_stopping_rounds, cv)
    elif cnfg.model_type == "regression":
        return benchmark_xgboost_regression(X, y, test_size, n_estimators, early_stopping_rounds, cv)
def benchmark_xgboost_regression(X, y, test_size=0.2, n_estimators=100, early_stopping_rounds=10, cv=5, random_state=42):
    """
    Runs an XGBoost regression benchmark model.

    Parameters:
    - X (pd.DataFrame or np.array): Feature matrix.
    - y (pd.Series or np.array): Target variable.
    - test_size (float): Proportion of data for the test set.
    - n_estimators (int): Number of boosting rounds.
    - early_stopping_rounds (int): Stops training if validation score doesn't improve.
    - cv (int): Number of cross-validation folds.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - model (XGBRegressor): Trained XGBoost model.
    - test_rmse (float): Root Mean Squared Error on the test set.
    - cv_rmse (float): Cross-validation RMSE score.
    """
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize features (optional but improves stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        objective="reg:squarederror",  # Regression objective
        eval_metric="rmse",
        tree_method="hist",  # Efficient histogram-based splits
        # early_stopping_rounds=early_stopping_rounds,
        random_state=random_state
    )

    # Train with early stopping
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-cv_scores.mean())

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Cross-Validation RMSE: {cv_rmse:.4f}")

    return model, test_rmse, cv_rmse


def benchmark_xgboost_classification(X, y, test_size=0.2, n_estimators=100, early_stopping_rounds=10, cv=5, random_state=42):
    """
    Runs an XGBoost classification benchmark model.

    Parameters:
    - X (pd.DataFrame or np.array): Feature matrix.
    - y (pd.Series or np.array): Target variable.
    - test_size (float): Proportion of data for the test set.
    - n_estimators (int): Number of boosting rounds.
    - early_stopping_rounds (int): Stops training if validation score doesn't improve.
    - cv (int): Number of cross-validation folds.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - model (XGBClassifier): Trained XGBoost model.
    - test_f1 (float): F1 score on the test set.
    - cv_f1 (float): Cross-validation F1 score.
    """
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize features (optional but improves stability)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",  # Binary classification objective
        eval_metric="logloss",# Logloss is commonly used for classification
        # early_stopping_rounds=early_stopping_rounds,
        random_state=random_state
    )

    # Train with early stopping
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_pred)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
    cv_f1 = cv_scores.mean()

    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Cross-Validation F1 Score: {cv_f1:.4f}")

    return model, test_f1, cv_f1
