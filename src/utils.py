import time

from algo import regression_methods_dict, classification_methods_dict


def time_function(func, *args, **kwargs):
    """
    Measures the execution time of a given function.

    Parameters:
        func (callable): The function to be timed.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: A tuple containing the result of the function and the elapsed time in seconds.
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def format_duration(seconds):
    """
    Formats a duration given in seconds into a human-readable string.

    Parameters:
        seconds (float): The duration in seconds.

    Returns:
        str: A formatted string in the format "X hours, Y minutes, Z seconds".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours} hours, {minutes} minutes, {secs} seconds"


def rename_duplicates(lst):
    """
    Renames duplicate items in a list by appending a counter to each duplicate.

    Parameters:
        lst (list): A list of items.

    Returns:
        list: A new list with duplicates renamed to include a counter.
    """
    counter = {}
    result = []
    for item in lst:
        counter[item] = counter.get(item, 0) + 1
        result.append(f"{item}_{counter[item]}")

    return result


def add_count_label(lst, lst_count):
    """
    Adds a count label to each item in a list.

    Parameters:
        lst (list): A list of items.
        lst_count (list): A list of counts corresponding to the items in `lst`.

    Returns:
        list: A new list with each item labeled with its count.
    """
    result = []
    for i in range(len(lst)):
        result.append(f"{lst[i]}\n{lst_count[i]} features")

    return result


def get_methods_dict(model_type):
    """
    Retrieves the appropriate methods dictionary based on the model type.

    Parameters:
        model_type (str): The type of model, either "classification" or "regression".

    Returns:
        dict: The methods dictionary corresponding to the model type.

    Raises:
        ValueError: If the model type is invalid.
    """
    if model_type == "classification":
        return classification_methods_dict
    elif model_type == "regression":
        return regression_methods_dict
    else:
        raise ValueError(f"Invalid model type: {model_type}")
