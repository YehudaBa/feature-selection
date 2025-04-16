import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, RFE

from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import config as cnfg


import  time_complecities as ts

def remove_zero_variance(X,y, threshold=1e-5, max_features=None):
    """
    Removes columns with variance close to zero from a Pandas DataFrame.

    Parameters:
        X (pd.DataFrame): The input DataFrame.
        threshold (float): Variance threshold below which columns are dropped.

    Returns:
        pd.DataFrame: The DataFrame with low-variance columns removed.
    """
    variances = X.var()  # Compute variance for each column
    return X.loc[:, variances > threshold]  # Keep columns with variance above the threshold




def remove_low_cv_features(X, y=None, threshold=0.1, epsilon=1e-8, min_features=1, max_features=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    cv_values = feature_stds / (feature_means + epsilon)
    selected_features = cv_values > threshold
    num_features_to_keep = max(min_features, max(X.shape[1] // 2, selected_features.sum()))
    if max_features:
        num_features_to_keep = min(num_features_to_keep, max_features)
    top_features = cv_values.nlargest(num_features_to_keep).index
    X_reduced = X.loc[:, top_features]
    print(f"Original feature count: {X.shape[1]}")
    print(f"Selected feature count: {X_reduced.shape[1]}")
    print(f"Removed {X.shape[1] - X_reduced.shape[1]} low-CV features (CV < {threshold}).")
    return X_reduced

def remove_majority_class_features(X, y=None, threshold=0.95, min_features=1, max_features=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    majority_proportion = X.apply(lambda col: col.value_counts(normalize=True).values[0], axis=0)
    selected_features = majority_proportion < threshold
    num_features_to_keep = max(min_features, max(X.shape[1] // 2, selected_features.sum()))
    if max_features:
        num_features_to_keep = min(num_features_to_keep, max_features)
    top_features = majority_proportion.nsmallest(num_features_to_keep).index
    X_reduced = X.loc[:, top_features]
    print(f"Original feature count: {X.shape[1]}")
    print(f"Selected feature count: {X_reduced.shape[1]}")
    print(f"Removed {X.shape[1] - X_reduced.shape[1]} features with majority proportion > {threshold}.")
    return X_reduced

def select_features_by_mi_threshold(X, y, threshold=0.95, min_features=1, max_features=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    cumulative_mi = mi_series.cumsum() / mi_series.sum()
    selected_features = cumulative_mi[cumulative_mi <= threshold].index
    num_features_to_keep = max(min_features, max(X.shape[1] // 2, len(selected_features)))
    if max_features:
        num_features_to_keep = min(num_features_to_keep, max_features)
    selected_features = mi_series.nlargest(num_features_to_keep).index
    X_reduced = X[selected_features]
    print(f"Original feature count: {X.shape[1]}")
    print(f"Selected feature count: {X_reduced.shape[1]}")
    print(f"Selected features contribute {threshold*100:.1f}% of total MI.")
    return X_reduced

def find_best_alpha(X, y, alphas=None, cv=5):
    """
    Automatically selects the best alpha for Lasso using cross-validation.

    Parameters:
    X (pd.DataFrame or np.array): Feature matrix
    y (pd.Series or np.array): Target variable
    alphas (list or np.array): List of alpha values to test (default: log scale)
    cv (int): Number of cross-validation folds

    Returns:
    float: Best alpha value
    np.array: Lasso coefficients at best alpha
    """
    if alphas is None:
        alphas = np.logspace(-0.5, 2, 50)  # 50 values from 0.0001 to 10

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lasso with Cross-Validation
    lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, n_jobs=-1)
    lasso_cv.fit(X_scaled, y)

    print(f"Best alpha: {lasso_cv.alpha_:.6f}")
    print(f"Number of selected features: {(lasso_cv.coef_ != 0).sum()} out of {X.shape[1]}")

    return lasso_cv.alpha_, lasso_cv.coef_


def correlation_based_feature_selection(X, y=None, threshold=0.98, min_features=1, max_features=None):
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    max_removal = max(X.shape[1] // 2, X.shape[1] - min_features)
    to_drop = to_drop[:max_removal]
    X_reduced = X.drop(columns=to_drop)
    if max_features:
        keep_features = X_reduced.columns[:max_features]
        X_reduced = X_reduced[keep_features]
    print(f"Original features: {X.shape[1]}")
    print(f"Removed features: {len(to_drop)} (max allowed: {max_removal})")
    print(f"Remaining features: {X_reduced.shape[1]}")
    return X_reduced

def lasso_feature_selection(X, y, alpha=0.001, min_features=1, max_features=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_scaled, y)

    selected_features = X.columns[lasso.coef_ != 0]
    max_removal = max(X.shape[1] // 2, X.shape[1] - min_features)

    if len(selected_features) < X.shape[1] - max_removal:
        selected_features = X.columns[np.argsort(np.abs(lasso.coef_))[-max_removal:]]

    if max_features is not None and len(selected_features) > max_features:
        importance = np.abs(lasso.coef_)
        selected_indices = np.argsort(importance)[-max_features:]
        selected_features = X.columns[selected_indices]

    X_reduced = X[selected_features]

    print(f"Alpha: {alpha}")
    print(f"Selected features: {len(selected_features)} out of {X.shape[1]} (max removal: {max_removal})")

    return X_reduced
# ToDo: change the absolute_num_of_features=cnfg.k_features to self.k_features (based on sqrt(n)
def random_forest_reg_feature_selection(X, y, n_estimators=300, absolute_num_of_features=None, required_percent=None, min_features=50, max_features=None):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]
    total_features = len(sorted_importances)
    percent_based_count = int(total_features * required_percent) if required_percent is not None else 0
    absolute_based_count = absolute_num_of_features if absolute_num_of_features is not None else 0
    num_features_to_keep = max(min_features, max(percent_based_count, absolute_based_count, total_features // 2))
    if max_features is not None:
        num_features_to_keep = min(num_features_to_keep, max_features)
    num_features_to_keep = max(1, min(num_features_to_keep, total_features))
    selected_features = sorted_features[:num_features_to_keep]

    print(f"Selected features: {len(selected_features)} out of {total_features} ({(len(selected_features) / total_features) * 100:.1f}%)")

    return X[selected_features]


from sklearn.ensemble import RandomForestClassifier
import numpy as np


def random_forest_clf_feature_selection(X, y, n_estimators=300,
                                        absolute_num_of_features=None,
                                        required_percent=None,
                                        min_features=50,
                                        max_features=None):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    total_features = len(sorted_importances)
    percent_based_count = int(total_features * required_percent) if required_percent is not None else 0
    absolute_based_count = absolute_num_of_features if absolute_num_of_features is not None else 0

    num_features_to_keep = max(min_features, max(percent_based_count, absolute_based_count, total_features // 2))
    if max_features is not None:
        num_features_to_keep = min(num_features_to_keep, max_features)
    num_features_to_keep = max(1, min(num_features_to_keep, total_features))

    selected_features = sorted_features[:num_features_to_keep]

    print(
        f"Selected features: {len(selected_features)} out of {total_features} ({(len(selected_features) / total_features) * 100:.1f}%)")

    return X[selected_features]


def rfe_feature_selection(X, y, max_features=100, model=None):
    """
    Perform Recursive Feature Elimination (RFE) to select features, ensuring at most half of the original features are removed.

    Parameters:
    - X (pd.DataFrame): Feature matrix
    - y (pd.Series or np.array): Target variable
    - max_features (int): Maximum number of features to keep
    - model (sklearn estimator, optional): Model to use for feature ranking (default: RandomForestRegressor)

    Returns:
    - pd.DataFrame: Reduced feature set
    - list: Selected feature names
    """
    # Default model: Random Forest
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Ensure at least n/2 features are selected
    min_features = max(X.shape[1] // 2, 1)
    n_features_to_select = min(max_features, X.shape[1])  # Cap at total feature count
    n_features_to_select = max(n_features_to_select, min_features)  # Ensure at least n/2 are selected

    # Apply RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    X_selected = rfe.fit_transform(X, y)

    # Get selected feature names
    selected_features = X.columns[rfe.support_]
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")

    return pd.DataFrame(X_selected, columns=selected_features)


regression_methods_dict = {#"rfe_feature_selection": {"pointer": rfe_feature_selection, "time_complexity": ts.O_kn_d2},
                        "remove_zero_variance": {"pointer": remove_zero_variance, "time_complexity": ts.O_nd, "index_method": "max_time_indexed_methods"},
                        "random_forest_feature_selection": {"pointer": random_forest_reg_feature_selection, "time_complexity": ts.O_nd_log_d, "index_method": "max_time_non_indexed_methods"},
                        "lasso_feature_selection": {"pointer": lasso_feature_selection, "time_complexity": ts.O_nd, "index_method": "max_time_non_indexed_methods"},
                        "correlation_based_feature_selection": {"pointer": correlation_based_feature_selection, "time_complexity": ts.O_d2, "index_method": "max_time_indexed_methods"},
                        "select_features_by_mi_threshold": {"pointer": select_features_by_mi_threshold, "time_complexity": ts.O_nd_log_n, "index_method": "max_time_indexed_methods"},
                        "remove_majority_class_features": {"pointer": remove_majority_class_features, "time_complexity": ts.O_nd, "index_method": "max_time_indexed_methods"},
                        "remove_low_cv_features": {"pointer": remove_low_cv_features, "time_complexity": ts.O_nd, "index_method": "max_time_indexed_methods"},
                        }


classification_methods_dict = {#"rfe_feature_selection": {"pointer": rfe_feature_selection, "time_complexity": ts.O_kn_d2},
                        "remove_zero_variance": {"pointer": remove_zero_variance, "time_complexity": ts.O_nd, "index_method": "max_time_indexed_methods"},
                        "random_forest_feature_selection": {"pointer": random_forest_clf_feature_selection, "time_complexity": ts.O_nd_log_d, "index_method": "max_time_non_indexed_methods"},
                        "lasso_feature_selection": {"pointer": lasso_feature_selection, "time_complexity": ts.O_nd, "index_method": "max_time_non_indexed_methods"},
                        "correlation_based_feature_selection": {"pointer": correlation_based_feature_selection, "time_complexity": ts.O_d2, "index_method": "max_time_indexed_methods"},
                        "select_features_by_mi_threshold": {"pointer": select_features_by_mi_threshold, "time_complexity": ts.O_nd_log_n, "index_method": "max_time_indexed_methods"},
                        "remove_majority_class_features": {"pointer": remove_majority_class_features, "time_complexity": ts.O_nd, "index_method": "max_time_indexed_methods"},
                        "remove_low_cv_features": {"pointer": remove_low_cv_features, "time_complexity": ts.O_nd, "index_method": "max_time_indexed_methods"},
                        }



