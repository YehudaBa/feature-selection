import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from numpy.ma.core import append

from modeling import benchmark_xgboost
import config as cnfg
from algo import regression_methods_dict, classification_methods_dict

if cnfg.model_type == "classification":
    methods_dict = classification_methods_dict.copy()
elif cnfg.model_type == "regression":
    methods_dict = regression_methods_dict.copy()


class FeatureSelection():
    def __init__(self):
        self.k_features  = cnfg.k_features
        self.dims = []
        self.used_methods = []

        # ToDo: move it to dict in config somehow

        self.methods = methods_dict.copy()


    def run(self):
        self.get_date()

        if self.k_features is None:
            self.k_features = np.round(np.sqrt(len(self.X)))
        self.dims.append(self.X.shape)
        if cnfg.anomaly_detection:
            self.methods.remove("remove_zero_variance")


        tmp_dim = self.X.shape[1]
        self.apply_methods()
        while tmp_dim != self.X.shape[1]:
            tmp_dim = self.X.shape[1]
            self.methods = methods_dict.copy()
            self.apply_methods()
        if self.X.shape[1] > self.k_features:
            self.methods = methods_dict.copy()
            best_method = self.benchmark_model()
            self.benchmarks = (len(self.used_methods)+1)*[0]+[1]
            self.used_methods.append(best_method)
            # ToDo: at this point, only once, and force it the number of required features k
            self.X = self.methods[best_method]["pointer"](self.X, self.y, max_features = 50)
            self.update_dims()
            del self.methods[best_method]



            a = 1

        print(self.dims)
        self.plot_feature_selection()

    def get_date(self):
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cnfg.input_path)
        self.df_data = pd.read_csv(input_path)
        self.X = self.df_data.drop(cnfg.index_cols + cnfg.drop_cols + [cnfg.target_col], axis = 1)
        self.y = self.df_data[cnfg.target_col]


    def select_next_method(self):
        if len(self.df_data) <= self.k_features:
            return None
        if not cnfg.anomaly_detection:
            pass

    def update_dims(self):
        self.dims.append(self.X.shape)

    def validate_time_complexity(self, method):
        if self.methods[method]["time_complexity"](*self.dims[-1]) <= cnfg.time_complexities[self.methods[method]["index_method"]]:
            return True



    def plot_feature_selection(self):
        """
        Plots the feature selection process:
        - X-axis: Feature selection methods used
        - Left Y-axis: Remaining number of features (dims)
        - Right Y-axis: Time complexity (tcs)
        """
        methods = ["orig_dim"]+self.used_methods
        dims = [x[1] for x in self.dims]
        tcs = [0]
        for i in range(len(self.used_methods)):
            tcs.append(methods_dict[self.used_methods[i]]["time_complexity"](*self.dims[i]))
        tcs = [tcs[i]/dims[i] for i in range(len(dims))]
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
        plt.title("Feature Selection Progression")
        plt.grid(True, linestyle="--", alpha=0.6)
        ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        # Show the plot
        plt.show()

    def apply_methods(self):

        for method in self.methods.keys():
            if self.X.shape[1] <= self.k_features:
                return
            if (self.dims[-1][0]*self.dims[-1][1]*np.log(self.dims[-1][0]) <= cnfg.benchmark_model_max_time) &(self.dims[-1][1] < self.dims[0][1]/4):

                return
            print(f"Applying {method}")
            if self.validate_time_complexity(method):
                self.used_methods.append(method)
                self.X = self.methods[method]["pointer"](self.X, self.y)
                self.update_dims()
                del self.methods[method]

                self.apply_methods()  # Restart from the first method
                return

    def benchmark_model(self):
        #self.methods = methods_dict.copy()
        methods_cost = {}
        best_key = None
        for method in [x for x in self.methods.keys() if x != "remove_zero_variance"]:
            print(f"Applying Benchmark Model {method}")
            if self.validate_time_complexity(method):
                #self.used_methods.append(method)
                X = self.methods[method]["pointer"](self.X, self.y)
                _, methods_cost[method], _ = benchmark_xgboost(X, self.y.copy())
                if cnfg.model_type == "classification":
                    best_key = min(methods_cost, key=methods_cost.get)
                else:
                    best_key = max(methods_cost, key=methods_cost.get)
                best_value = methods_cost[best_key]
                # ToDo: show min_value
        return  best_key



def rename_duplicates(lst):
    counter = {}
    result = []

    for item in lst:
        counter[item] = counter.get(item, 0) + 1
        result.append(f"{item}_{counter[item]}")

    return result

def add_count_label(lst, lst_count):
    result = []
    for i in range(len(lst)):
        result.append(f"{lst[i]}\n{lst_count[i]} features")

    return result




if __name__ == "__main__":
    fs = FeatureSelection()
    fs.run()





    a = 1
