import json
import math
import os
import uuid

import matplotlib.pyplot as plt
import pandas as pd

import config as cnfg
from modeling import benchmark_xgboost
from time_complexities import O_ndT
from utils import rename_duplicates, add_count_label, get_methods_dict, time_function, format_duration


class FeatureSelection():
    """
    A class to perform feature selection using various unsupervised and supervised methods.

    Attributes:
        dims (list): Stores the dimensions of the feature matrix at different stages.
        used_methods (list): Stores the names of methods applied during feature selection.
        orig_methods (dict): Stores the original methods dictionary based on the model type.
    """

    def __init__(self):
        self.dims = []
        self.used_methods = []
        self.orig_methods = get_methods_dict(cnfg.model_type)

    def setup_parameters(self):
        """
        Sets up the parameters for feature selection based on the configuration.
        """
        if cnfg.anomaly_detection:
            self.methods.remove("remove_zero_variance")
        self.methods = self.orig_methods.copy()
        self.run_id = uuid.uuid4().hex

    def get_date(self):
        """
        Loads the dataset and splits it into features (X) and target (y) based on the configuration.
        """
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cnfg.input_path)
        self.df_data = pd.read_csv(input_path)
        self.X = self.df_data.drop(cnfg.index_cols + cnfg.drop_cols + [cnfg.target_col], axis=1)
        self.y = self.df_data[cnfg.target_col]

    def set_k_features(self):
        """
        Sets the target number of features (k_features) based on the configuration.
        Ensures only one of `k_features` or `percent_features` is provided.
        """
        if cnfg.k_features is not None and cnfg.percent_features is not None:
            raise ValueError("Only one of k_features or percent_features should be provided.")
        elif cnfg.k_features is not None:
            self.k_features = cnfg.k_features
        elif cnfg.percent_features is not None:
            self.k_features = int(cnfg.percent_features * self.X.shape[1])
        else:
            self.k_features = int(math.sqrt(self.X.shape[0]))

    def update_dims(self):
        """
        Updates the dimensions list with the current shape of the feature matrix.
        """
        self.dims.append(self.X.shape)

    def validate_time_complexity(self, method):
        """
        Validates if the time complexity of a method is within the allowed limit.

        Parameters:
            method (str): The name of the method to validate.

        Returns:
            bool: True if the method's time complexity is valid, False otherwise.
        """
        if self.methods[method]["time_complexity"](*self.dims[-1]) <= cnfg.time_complexities[
            self.methods[method]["index_method"]]:
            return True

    def is_reduction_goal_met(self):
        """
        Checks if the feature reduction goal has been met.

        Returns:
            bool: True if the reduction goal is met, False otherwise.
        """
        if self.X.shape[1] <= self.k_features:
            print(f"Reduction goal met, stopping feature selection.")
            return True
        return False

    def can_run_benchmark_model(self):
        """
        Checks if the benchmark model can be run based on time complexity and feature reduction:
        - dimensions reduced enough for time complexity
        - least 4 models run before benchmark (if they could)
        Returns:
            bool: True if the benchmark model can be run, False otherwise.
        """
        tc = O_ndT(self.dims[-1][0], self.dims[-1][1])
        if (tc <= cnfg.benchmark_model_max_time) & (
                self.dims[-1][1] < self.dims[0][1] / 4):
            return True
        else:
            return False

    def apply_methods(self):
        """
        Applies feature selection methods iteratively until the reduction goal is met
        or the benchmark model can be run.
        """
        for method in self.methods.keys():
            if self.is_reduction_goal_met():
                return
            if self.can_run_benchmark_model():
                return
            if self.validate_time_complexity(method):
                print(f"Applying {method}")
                self.used_methods.append(method)
                self.X = self.methods[method]["pointer"](self.X, self.y, min_features=self.k_features)
                self.update_dims()
                del self.methods[method]
                self.apply_methods()
                return

    def benchmark_model(self):
        """
        Runs the benchmark model to determine the best feature selection method.

        Returns:
            str: The name of the best feature selection method.
        """
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
        """
        Runs unsupervised feature selection methods iteratively until no further reduction is achieved.
        """
        tmp_dim = self.X.shape[1]
        self.apply_methods()
        while tmp_dim != self.X.shape[1]:
            tmp_dim = self.X.shape[1]
            self.methods = self.orig_methods.copy()
            self.apply_methods()

    def run_supervised_selections(self):
        """
        Runs supervised feature selection methods using the benchmark model.
        """
        if self.is_reduction_goal_met():
            return
        self.methods = self.orig_methods.copy()
        best_method = self.benchmark_model()
        self.benchmarks = (len(self.used_methods) + 1) * [0] + [1]
        self.used_methods.append(best_method)
        print(f"Applying Benchmark Model {best_method}")
        self.X = self.methods[best_method]["pointer"](self.X, self.y, min_features=self.k_features,
                                                      max_features=self.k_features)
        self.update_dims()
        del self.methods[best_method]

    def plot_feature_selection(self):
        """
        Plots the progression of feature selection, showing the number of features and time complexity.

        Saves the plot as a PNG file with a unique name.
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
        unique_filename = f"feature_selection_{self.run_id}.png"
        # Save the plot to the file
        plt.savefig(unique_filename, dpi=300, bbox_inches='tight')

    def save_selected_features(self):
        '''
        Save the selected features into txt file with the same run_id as in the plot file name
        :return:  None
        '''
        print(f"`Selected Features:\n {list(self.X.columns)}")
        with open(f"best_features_{self.run_id}", "w") as file:
            json.dump(list(self.X.columns), file)

    def pipeline(self):
        """
        Executes the entire feature selection pipeline, including parameter setup,
        data loading, unsupervised and supervised selections.
        """
        self.setup_parameters()
        self.get_date()
        self.set_k_features()
        self.dims.append(self.X.shape)
        self.run_unsupervised_selections()
        self.run_supervised_selections()

    def run(self):
        """
        Runs the feature selection pipeline and plots the results.

        Measures the total runtime of the pipeline.
        """
        _, self.run_time = time_function(self.pipeline)
        self.plot_feature_selection()
        self.save_selected_features()
        print(f"\nThe results saved to {self.run_id}.png and\n {self.run_id}.txt")
