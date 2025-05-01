from algo import regression_methods_dict, classification_methods_dict
# ToDo: refactor
def rename_duplicates(lst):
    counter = {}
    result = []

    for item in lst:
        counter[item] = counter.get(item, 0) + 1
        result.append(f"{item}_{counter[item]}")

    return result
# ToDo: refactor
def add_count_label(lst, lst_count):
    result = []
    for i in range(len(lst)):
        result.append(f"{lst[i]}\n{lst_count[i]} features")

    return result
# ToDo: refactor
def get_methods_dict(model_type):
    if model_type == "classification":
        return classification_methods_dict
    elif model_type == "regression":
        return regression_methods_dict
    else:
        raise ValueError(f"Invalid model type: {model_type}")

