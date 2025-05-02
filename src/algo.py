import numpy as np
import pandas as pd
import time_complexities as ts
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier




def remove_zero_variance(X, y, threshold=1e-5, min_features=50, max_features=None):
    """
    Removes columns with variance close to zero from a Pandas DataFrame.

    Parameters:
        X (pd.DataFrame): The input DataFrame.
        threshold (float): Variance threshold below which columns are dropped.
    min_features and max_features are just the stucture, 
    Returns:
        pd.DataFrame: The DataFrame with low-variance columns removed.
    """
    variances = X.var()  # Compute variance for each column
    return X.loc[:, variances > threshold]  # Keep columns with variance above the threshold



def remove_low_cv_features(X, y=None, threshold=0.1, epsilon=1e-8, min_features=50,
                           max_features=None, min_retention_ratio=0.5, threshold_decay=0.9):
    """
    Removes features with low coefficient of variation (CV) from a dataset,
    ensuring that no more than 50% of features are removed.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input feature matrix.
    y : Ignored
        Included for compatibility with scikit-learn-style function signatures.
    threshold : float, optional
        Initial CV threshold to remove low-variation features.
    epsilon : float, optional
        Small value added to mean to avoid division by zero.
    min_features : int, optional
        Minimum number of features to keep.
    max_features : int or None, optional
        Maximum number of features to keep.
    min_retention_ratio : float, optional
        Minimum ratio of features to retain (default is 0.5).
    threshold_decay : float, optional
        Decay factor for threshold if too many features are removed.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with selected features.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    n_total_features = X.shape[1]
    min_allowed_features = max(int(n_total_features * min_retention_ratio), min_features)

    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    cv_values = feature_stds / (feature_means + epsilon)

    current_threshold = threshold

    while True:
        selected = cv_values > current_threshold
        n_selected = selected.sum()

        if n_selected >= min_allowed_features or current_threshold < 1e-6:
            break
        else:
            current_threshold *= threshold_decay

    # Ensure at least `min_features`
    if n_selected < min_features:
        top_indices = cv_values.nlargest(min_features).index
    else:
        top_indices = cv_values[selected].nlargest(n_selected).index

    # Apply `max_features` limit
    if max_features is not None and len(top_indices) > max_features:
        top_indices = cv_values[top_indices].nlargest(max_features).index

    X_reduced = X[top_indices]

    print(f"Final CV threshold used: {current_threshold:.6f}")
    print(f"Original feature count: {n_total_features}")
    print(f"Selected feature count: {X_reduced.shape[1]}")
    print(f"Removed {n_total_features - X_reduced.shape[1]} low-CV features.")

    return X_reduced

def remove_majority_class_features(X, y=None, threshold=0.95, min_features=50,
                                   max_features=None, min_retention_ratio=0.5, threshold_decay=0.95):
    """
    Removes features where a single value dominates, ensuring at least half of the features are retained.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input feature matrix.
    y : Ignored
        Included for compatibility with scikit-learn-style function signatures.
    threshold : float, optional
        Initial threshold for the maximum allowed proportion of the most frequent value.
    min_features : int, optional
        Minimum number of features to retain.
    max_features : int or None, optional
        Maximum number of features to retain.
    min_retention_ratio : float, optional
        Minimum ratio of features to keep (default is 0.5).
    threshold_decay : float, optional
        Factor to reduce threshold by if too many features are removed.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing selected features.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    n_total_features = X.shape[1]
    min_allowed_features = max(int(n_total_features * min_retention_ratio), min_features)

    majority_proportion = X.apply(lambda col: col.value_counts(normalize=True).values[0], axis=0)
    current_threshold = threshold

    while True:
        selected = majority_proportion < current_threshold
        n_selected = selected.sum()

        if n_selected >= min_allowed_features or current_threshold < 1e-6:
            break
        else:
            current_threshold *= threshold_decay

    # Ensure at least `min_features`
    if n_selected < min_features:
        top_indices = majority_proportion.nsmallest(min_features).index
    else:
        top_indices = majority_proportion[selected].nsmallest(n_selected).index

    # Apply `max_features` limit
    if max_features is not None and len(top_indices) > max_features:
        top_indices = majority_proportion[top_indices].nsmallest(max_features).index

    X_reduced = X[top_indices]

    print(f"Final threshold used: {current_threshold:.6f}")
    print(f"Original feature count: {n_total_features}")
    print(f"Selected feature count: {X_reduced.shape[1]}")
    print(f"Removed {n_total_features - X_reduced.shape[1]} majority-class features.")

    return X_reduced

def select_features_by_mi_threshold(X, y, threshold=0.95, min_features=50,
                                    max_features=None, min_retention_ratio=0.5,
                                    threshold_decay=0.95):
    """
    Selects features based on their mutual information (MI) with the target,
    while ensuring no more than 50% of features are removed.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input feature matrix.
    y : array-like
        Target variable.
    threshold : float
        Initial cumulative MI threshold for feature selection.
    min_features : int
        Minimum number of features to retain.
    max_features : int or None
        Maximum number of features to retain.
    min_retention_ratio : float
        Minimum proportion of original features to retain.
    threshold_decay : float
        Decay factor for threshold if too many features are removed.

    Returns:
    --------
    pandas.DataFrame
        Reduced DataFrame with selected features.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    n_total_features = X.shape[1]
    min_allowed_features = max(int(n_total_features * min_retention_ratio), min_features)

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    current_threshold = threshold

    while True:
        cumulative_mi = mi_series.cumsum() / mi_series.sum()
        selected = cumulative_mi[cumulative_mi <= current_threshold].index
        n_selected = len(selected)

        if n_selected >= min_allowed_features or current_threshold < 1e-6:
            break
        else:
            current_threshold *= threshold_decay

    # Ensure at least `min_features`
    if n_selected < min_features:
        selected = mi_series.nlargest(min_features).index
    else:
        selected = mi_series[selected].index

    # Apply `max_features` limit
    if max_features is not None and len(selected) > max_features:
        selected = mi_series[selected].nlargest(max_features).index

    X_reduced = X[selected]

    print(f"Final MI threshold used: {current_threshold:.6f}")
    print(f"Original feature count: {n_total_features}")
    print(f"Selected feature count: {X_reduced.shape[1]}")
    print(f"Selected features contribute at least {current_threshold * 100:.1f}% of total MI.")

    return X_reduced


def correlation_based_feature_selection(X, y=None, threshold=0.98, min_features=50,
                                        max_features=None, min_retention_ratio=0.5,
                                        threshold_decay=0.98):
    """
    Removes highly correlated features, adaptively adjusting the threshold to retain enough features.

    Parameters:
    -----------
    X : pandas.DataFrame
        Input feature matrix. Assumes numerical data.
    y : Ignored
        Included for compatibility with scikit-learn-style function signatures.
    threshold : float
        Initial correlation threshold above which one of the correlated feature pairs is dropped.
    min_features : int
        Minimum number of features to retain.
    max_features : int or None
        Maximum number of features to retain.
    min_retention_ratio : float
        Minimum proportion of features to retain (default 0.5).
    threshold_decay : float
        Multiplicative factor to reduce threshold if too many features are dropped.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with reduced features after removing highly correlated ones.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    n_total = X.shape[1]
    min_allowed = max(int(n_total * min_retention_ratio), min_features)
    current_threshold = threshold

    while True:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > current_threshold)]
        keep_count = n_total - len(to_drop)

        if keep_count >= min_allowed or current_threshold <= 0:
            break
        current_threshold *= threshold_decay

    # Actually drop selected features (limited by how many we’re allowed to drop)
    max_removal = n_total - min_allowed
    to_drop = to_drop[:max_removal]
    X_reduced = X.drop(columns=to_drop)

    # Enforce max_features limit
    if max_features is not None and X_reduced.shape[1] > max_features:
        X_reduced = X_reduced.iloc[:, :max_features]

    print(f"Final correlation threshold used: {current_threshold:.4f}")
    print(f"Original features: {n_total}")
    print(f"Removed features: {len(to_drop)} (max allowed: {max_removal})")
    print(f"Remaining features: {X_reduced.shape[1]}")

    return X_reduced

def lasso_feature_selection(X, y, alpha=0.001, min_features=50, max_features=None, min_retention_ratio=0.5, alpha_decay=0.5):
    """
    Selects features using Lasso regression while ensuring no more than 50% of features are removed.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Input feature matrix.
    y : array-like
        Target variable.
    alpha : float, optional
        Initial regularization strength.
    min_features : int, optional
        Minimum number of features to retain.
    max_features : int or None, optional
        Maximum number of features to retain.
    min_retention_ratio : float, optional
        Minimum ratio of features to retain (e.g., 0.5 means keep at least 50%).
    alpha_decay : float, optional
        Factor to reduce alpha by if too many features are removed.

    Returns:
    --------
    pandas.DataFrame
        Reduced feature set.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    n_total_features = X.shape[1]
    min_allowed_features = max(int(n_total_features * min_retention_ratio), min_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    current_alpha = alpha

    while True:
        lasso = Lasso(alpha=current_alpha, random_state=42)
        lasso.fit(X_scaled, y)
        nonzero_coef_indices = np.flatnonzero(lasso.coef_)
        selected_features = X.columns[nonzero_coef_indices]

        if len(selected_features) >= min_allowed_features:
            break
        else:
            current_alpha *= alpha_decay
            if current_alpha < 1e-6:  # prevent infinite loop
                print("Alpha too small. Returning top features to satisfy constraints.")
                break

    # Ensure at least min_features
    if len(selected_features) < min_features:
        top_indices = np.argsort(np.abs(lasso.coef_))[-min_features:]
        selected_features = X.columns[top_indices]

    # Apply max_features limit
    if max_features is not None and len(selected_features) > max_features:
        importance = np.abs(lasso.coef_)
        selected_indices = np.argsort(importance)[-max_features:]
        selected_features = X.columns[selected_indices]

    X_reduced = X[selected_features]

    print(f"Final alpha used: {current_alpha}")
    print(f"Selected features: {len(selected_features)} out of {n_total_features} (min allowed: {min_allowed_features})")

    return X_reduced

def filter_features_random_forest(rf, X, min_retention_ratio, min_features, max_features):
    feature_importances = rf.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    sorted_features = X.columns[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]
    total_features = len(sorted_importances)
    n_total_features = X.shape[1]
    num_features_to_keep = max(int(n_total_features * min_retention_ratio), min_features)

    if max_features is not None:
        num_features_to_keep = min(num_features_to_keep, max_features)
    num_features_to_keep = max(1, min(num_features_to_keep, total_features))
    selected_features = sorted_features[:num_features_to_keep]

    print(
        f"Selected features: {len(selected_features)} out of {total_features} ({(len(selected_features) / total_features) * 100:.1f}%)")
    return X[selected_features]

def random_forest_reg_feature_selection(X, y, n_estimators=300, min_features=50, max_features=None, min_retention_ratio=0.5):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return filter_features_random_forest(rf, X, min_retention_ratio, min_features, max_features)


def random_forest_clf_feature_selection(X, y, n_estimators=300,
                                        min_features=50,
                                        max_features=None, min_retention_ratio=0.5):

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return filter_features_random_forest(rf, X, min_retention_ratio, min_features, max_features)



regression_methods_dict = {
    "remove_zero_variance": {"pointer": remove_zero_variance, "time_complexity": ts.O_nd,
                             "index_method": "max_time_indexed_methods"},
    "random_forest_feature_selection": {"pointer": random_forest_reg_feature_selection,
                                        "time_complexity": ts.O_nd_log_n,
                                        "index_method": "max_time_non_indexed_methods"},
    "lasso_feature_selection": {"pointer": lasso_feature_selection, "time_complexity": ts.O_nd,
                                "index_method": "max_time_non_indexed_methods"},
    "correlation_based_feature_selection": {"pointer": correlation_based_feature_selection, "time_complexity": ts.O_d2,
                                            "index_method": "max_time_indexed_methods"},
    "select_features_by_mi_threshold": {"pointer": select_features_by_mi_threshold, "time_complexity": ts.O_nd_log_n,
                                        "index_method": "max_time_indexed_methods"},
    "remove_majority_class_features": {"pointer": remove_majority_class_features, "time_complexity": ts.O_nd,
                                       "index_method": "max_time_indexed_methods"},
    "remove_low_cv_features": {"pointer": remove_low_cv_features, "time_complexity": ts.O_nd,
                               "index_method": "max_time_indexed_methods"},
}

classification_methods_dict = {
    "remove_zero_variance": {"pointer": remove_zero_variance, "time_complexity": ts.O_nd,
                             "index_method": "max_time_indexed_methods"},
    "random_forest_feature_selection": {"pointer": random_forest_clf_feature_selection,
                                        "time_complexity": ts.O_nd_log_n,
                                        "index_method": "max_time_non_indexed_methods"},
    "lasso_feature_selection": {"pointer": lasso_feature_selection, "time_complexity": ts.O_nd,
                                "index_method": "max_time_non_indexed_methods"},
    "correlation_based_feature_selection": {"pointer": correlation_based_feature_selection, "time_complexity": ts.O_d2,
                                            "index_method": "max_time_indexed_methods"},
    "select_features_by_mi_threshold": {"pointer": select_features_by_mi_threshold, "time_complexity": ts.O_nd_log_n,
                                        "index_method": "max_time_indexed_methods"},
    "remove_majority_class_features": {"pointer": remove_majority_class_features, "time_complexity": ts.O_nd,
                                       "index_method": "max_time_indexed_methods"},
    "remove_low_cv_features": {"pointer": remove_low_cv_features, "time_complexity": ts.O_nd,
                               "index_method": "max_time_indexed_methods"},
}
