import time
from algo import regression_methods_dict, classification_methods_dict

def time_function(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours} hours, {minutes} minutes, {secs} seconds"

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

def get_methods_dict(model_type):
    if model_type == "classification":
        return classification_methods_dict
    elif model_type == "regression":
        return regression_methods_dict
    else:
        raise ValueError(f"Invalid model type: {model_type}")

