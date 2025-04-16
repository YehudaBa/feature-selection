#input_path = "resources/data.csv"
input_path = "resources/rna_data/RNA_sample_processed.csv"
# required number of features in the output
k_features = 50

# target_col = "Lympho"
# drop_cols = ["ER"]
# index_cols = ["samplename"]

# target_col = "ER"
# drop_cols = ["Lympho"]
# index_cols = ["samplename"]

target_col = "status"
drop_cols = []
index_cols = ["Sample_id"]


thresholds = {}
# is it anomaly detection or not
anomaly_detection = False

# ToDo: set a default of minimum n*m
# max allowing time for indexed methods/ non indexed methods
time_complexities = {
"max_time_indexed_methods" : 94727754+1,
"max_time_non_indexed_methods" : 175798159 }#40000000}

benchmark_model_max_time = 3000000

model_type = "classification"
# model_type = "regression"
