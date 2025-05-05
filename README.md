# SmartSelect: Ensemble-Based Feature Selection for Machine Learning

A master's thesis project exploring an ensemble approach to feature selection, combining multiple methods to enhance model accuracy, robustness, and interpretability in machine learning tasks.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)


## Introduction
In modern machine learning workflows, selecting the most relevant features from high-dimensional data is crucial for building efficient, interpretable, and high-performing models. Feature selection helps reduce overfitting, improve generalization, and decrease computational cost — especially in domains where datasets contain many potentially redundant or irrelevant features.

This project, developed as part of a master's thesis, proposes an ensemble-based feature selection framework that leverages the strengths of multiple selection strategies — including statistical filters, wrapper methods, and embedded techniques. By combining diverse approaches, the ensemble aims to produce more stable and accurate feature subsets than any individual method alone.

The framework is designed to be modular, extensible, and applicable to both classification and regression problems. It includes tools for ranking, aggregating, and visualizing selected features across methods, with the goal of assisting data scientists in building better models with fewer, more meaningful inputs.

## Installation
ToDo

## Usage

### Input Format

The input data must be a csv file, and follow this structure:

- **Target column**: *(str)* A single column representing the prediction target.
- **Optional columns**:
  - **`drop_cols`** *(List[str])*: Columns that must be excluded from model training (e.g., identifiers, known leakage).
  - **`index_cols`** *(List[str])*: Columns used for indexing or metadata, not included in training or selection.
- All the other columns will be considered as features for selection.

> The model will automatically ignore all columns listed in `drop_cols` and `index_cols`.

---

### Configuration

This section defines all required and optional parameters for running the feature selection pipeline.

All configuration parameters are defined in the `config.py` file.

---

#### Input Location

- **`input_path`** *(str)*: Path to the input data file (e.g., CSV or Parquet).

---

#### Feature Count

Specify how many features to select:

- **`k_features`** *(int or None)*: Absolute number of features to select.
- **`percent_features`** *(float or None)*: Percentage (e.g., `0.2` for 20%) of features to select.

> ⚠️ **Exactly one** of `k_features` or `percent_features` must be `None`.  
If both are set, the model raises an error.  
If both are `None`, the default is `sqrt(n_samples)`.

---

#### Column Settings

- **`target_col`** *(str)*: Name of the target column.
- **`drop_cols`** *(List[str])*: List of forbidden feature columns.
- **`index_cols`** *(List[str])*: List of index/metadata columns.

---

#### Model Type

- **`model_type`** *(str)*: Type of task to solve — must be either `"classification"` or `"regression"`.
- **`Anomaly Detection (Not yet supported)`** *(bool)*: Currently support only False. Placeholder for future development.

---

#### Time Complexity Controls

To manage computational cost during feature selection:

```python
time_complexities = {
    "max_time_indexed_methods": int, # Indexed methods: Fast methods like correlation
    "max_time_non_indexed_methods": int # Non-indexed methods: Slower methods like Random Forest or RFE
}
```
- **`benchmark_model_max_time`** Maximum time cost allowed for the benchmark model, which runs once at the final stage. Multiplied by the number of ensemble models





---

