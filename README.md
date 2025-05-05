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

### Configuration:

All configuration parameters are defined in the `config.py` file.

#### Input Format

The input data must be in a tabular format and include:

- **Target column**: A single column containing the prediction target (classification or regression).
- **Optional columns**:
  - **Drop columns**: Columns that must not be used by the model (e.g., identifiers or data leakage sources).
  - **Index columns**: Columns used only for identification or metadata, not for training.
- All the other columns will be considered as features for selection.

---

