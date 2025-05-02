import os
import math
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cnfg
from modeling import benchmark_xgboost
from time_complexities import O_nd_log_n
from utils import rename_duplicates, add_count_label, get_methods_dict, time_function, format_duration


class FeatureSelection():
    def __init__(self):
        self.dims = []
        self.used_methods = []
        self.orig_methods = get_methods_dict(cnfg.model_type)

    def setup_parameters(self):
        if cnfg.anomaly_detection:
            self.methods.remove("remove_zero_variance")
        self.methods = self.orig_methods.copy()

    def get_date(self):
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cnfg.input_path)
        self.df_data = pd.read_csv(input_path)
        self.X = self.df_data.drop(cnfg.index_cols + cnfg.drop_cols + [cnfg.target_col], axis=1)
        self.y = self.df_data[cnfg.target_col]

    def set_k_features(self):
        if cnfg.k_features is not None and cnfg.percent_features is not None:
            raise ValueError("Only one of k_features or percent_features should be provided.")
        elif cnfg.k_features is not None:
            self.k_features = cnfg.k_features
        elif cnfg.percent_features is not None:
            self.k_features = int(cnfg.percent_features * self.X.shape[1])
        else:
            self.k_features = int(math.sqrt(self.X.shape[1]))


    def update_dims(self):
        self.dims.append(self.X.shape)

    def validate_time_complexity(self, method):
        if self.methods[method]["time_complexity"](*self.dims[-1]) <= cnfg.time_complexities[
            self.methods[method]["index_method"]]:
            return True

    def is_reduction_goal_met(self):

        if self.X.shape[1] <= self.k_features:
            print(f"Reduction goal met, stopping feature selection.")
            return True

    def can_run_benchmark_model(self):
        # check if we can already run the benchmark model
        # dimensions reduced enough for time complexity
        # and at least 4 models run before benchmark (if they could)
        tc = O_nd_log_n(self.dims[-1][0], self.dims[-1][1])
        if (tc <= cnfg.benchmark_model_max_time) & (
                self.dims[-1][1] < self.dims[0][1] / 4):
            return True
        else:
            return False

    def apply_methods(self):
        for method in self.methods.keys():
            if self.is_reduction_goal_met():
                return
            if self.can_run_benchmark_model():
                return
            if self.validate_time_complexity(method):
                print(f"Applying {method}")
                self.used_methods.append(method)
                self.X = self.methods[method]["pointer"](self.X, self.y, min_features = self.k_features)
                self.update_dims()
                del self.methods[method]
                self.apply_methods()
                return

    def benchmark_model(self):
        methods_cost = {}
        best_key = None
        for method in [x for x in self.methods.keys() if x != "remove_zero_variance"]:
            if self.validate_time_complexity(method):
                X = self.methods[method]["pointer"](self.X, self.y)
                _, methods_cost[method], _ = benchmark_xgboost(X, self.y.copy())
                if cnfg.model_type == "classification":
                    best_key = min(methods_cost, key=methods_cost.get)
                else:
                    best_key = max(methods_cost, key=methods_cost.get)
        return best_key

    def run_unsupervised_selections(self):
        tmp_dim = self.X.shape[1]
        self.apply_methods()
        while tmp_dim != self.X.shape[1]:
            tmp_dim = self.X.shape[1]
            self.methods = self.orig_methods.copy()
            self.apply_methods()

    def run_supervised_selections(self):
        if self.is_reduction_goal_met():
            return
        self.methods = self.orig_methods.copy()
        best_method = self.benchmark_model()
        self.benchmarks = (len(self.used_methods) + 1) * [0] + [1]
        self.used_methods.append(best_method)
        print(f"Applying Benchmark Model {best_method}")
        self.X = self.methods[best_method]["pointer"](self.X, self.y, min_features=self.k_features, max_features=self.k_features)
        self.update_dims()
        del self.methods[best_method]

    def plot_feature_selection(self):
        """
        Plots the feature selection process:
        - X-axis: Feature selection methods used
        - Left Y-axis: Remaining number of features (dims)
        - Right Y-axis: Time complexity (tcs)
        """
        methods = ["orig_dim"] + self.used_methods
        dims = [x[1] for x in self.dims]
        tcs = [0]
        for i in range(len(self.used_methods)):
            tcs.append(self.orig_methods[self.used_methods[i]]["time_complexity"](*self.dims[i]))
        tcs = [tcs[i] / dims[i] for i in range(len(dims))]
        methods = rename_duplicates(methods)
        methods = add_count_label(methods, dims)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot feature dimensions
        ax1.plot(methods, dims, marker='o', color='b', label="Remaining Features")
        ax1.set_xlabel("Feature Selection Methods", fontsize=5)
        ax1.set_ylabel("Number of Features", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot time complexity on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(methods, tcs, marker='s', color='r', linestyle='dashed', label="Time Complexity")
        ax2.set_ylabel("Time Complexity Index \n (normalized by dimensions)", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Rotate X-axis labels
        plt.xticks(rotation=90, ha="right", fontsize=5)
        plt.title(f"Feature Selection Progression\n{format_duration(self.run_time)}")

        plt.grid(True, linestyle="--", alpha=0.6)
        ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        # Generate a unique file name using UUID
        unique_filename = f"feature_selection_{uuid.uuid4().hex}.png"
        # Save the plot to the file
        plt.savefig(unique_filename, dpi=300, bbox_inches='tight')

    def pipeline(self):
        self.setup_parameters()
        self.get_date()
        self.set_k_features()
        self.dims.append(self.X.shape)
        self.run_unsupervised_selections()
        self.run_supervised_selections()

    def run(self):
        _, self.run_time = time_function(self.pipeline)
        self.plot_feature_selection()
