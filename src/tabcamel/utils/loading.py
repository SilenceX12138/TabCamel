from __future__ import annotations

import os
from typing import Optional

import bnlearn as bn
import numpy as np
import pandas as pd
import scipy.io as spio
from pgmpy.utils import get_example_model
from sklearn import datasets
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

from .. import (DUMMY_TARGET, dataset2bnlearn_id, dataset2openml_id, dataset2path, dataset2pgmpy_id, dataset2sklearn_id,
                dataset2uci_id)


def load_tabular_dataset(
    dataset_name: str,
    target_col: Optional[str] = None,
    metafeature_dict: Optional[dict] = None,
) -> dict:
    """Load a tabular dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        target_col (Optional[str], optional): Name of the target column.
        metafeature_dict (Optional[dict], optional): Dictionary containing the metafeatures.

    Raises:
        ValueError: If the dataset is not recognised.

    Returns:
        dict: Dictionary containing the features and target variables.
    """
    # ===== Load the dataset =====
    dataset_source = get_dataset_source(dataset_name)
    if dataset_source == "openml":
        dataset = load_openml_dataset(dataset_name)
    elif dataset_source == "uci":
        dataset = load_uci_dataset(dataset_name)
    elif dataset_source == "sklearn":
        dataset = load_sklearn_dataset(dataset_name)
    elif dataset_source == "bnlearn":
        dataset = load_bnlearn_dataset(dataset_name)
    elif dataset_source == "pgmpy":
        dataset = load_pgmpy_dataset(dataset_name)
    elif dataset_source == "local":
        dataset = load_local_dataset(dataset_name, target_col, metafeature_dict)
    else:
        raise ValueError(f"Dataset {dataset_name} not recognised.")

    return {
        "data_source": dataset_source,
        "data_id": dataset["id"],
        "data_dict": dataset["data_dict"],
    }


def get_dataset_source(dataset_name: str) -> str:
    """Get the source of dataset.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        str: Type of dataset (openml or local).
    """
    if dataset_name in dataset2openml_id.keys():
        dataset_type = "openml"
    elif dataset_name in dataset2uci_id.keys():
        dataset_type = "uci"
    elif dataset_name in dataset2sklearn_id.keys():
        dataset_type = "sklearn"
    elif dataset_name in dataset2bnlearn_id.keys():
        dataset_type = "bnlearn"
    elif dataset_name in dataset2pgmpy_id.keys():
        dataset_type = "pgmpy"
    elif os.path.exists(dataset_name) or dataset_name in dataset2path.keys():
        dataset_type = "local"
    else:
        raise ValueError(f"Dataset {dataset_name} not recognised.")

    return dataset_type


def load_openml_dataset(dataset_name: str) -> dict:
    """Load a dataset from OpenML.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dict: Dictionary containing the features and target variables.
    """
    # ===== Load the dataset with id=====
    dataset_id = dataset2openml_id[dataset_name]
    data_loaded = fetch_openml(data_id=dataset_id, as_frame=True)
    # ===== Get the features and target variables =====
    X_df = data_loaded.data
    y_s = data_loaded.target

    return {
        "id": dataset_id,
        "data_dict": {
            "X": X_df,
            "y": y_s,
        },
    }


def load_uci_dataset(dataset_name: str) -> dict:
    # ===== Load the dataset with id=====
    dataset_id = dataset2uci_id[dataset_name]
    data_loaded = fetch_ucirepo(id=dataset_id)
    # ===== Get the features and target variables =====
    X_df = data_loaded.data.features
    y_s = data_loaded.data.targets.iloc[:, 0]

    return {
        "id": dataset_id,
        "data_dict": {
            "X": X_df,
            "y": y_s,
        },
    }


def load_sklearn_dataset(dataset_name: str) -> dict:
    """Load a dataset from sklearn.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dict: Dictionary containing the features and target variables.
    """
    # ===== Load the dataset with id=====
    dataset_id = dataset2sklearn_id[dataset_name]
    data_loader = getattr(datasets, dataset_id)
    data_loaded = data_loader(as_frame=True)

    # ===== Get the features and target variables =====
    X_df = data_loaded.data
    y_s = data_loaded.target

    return {
        "id": dataset_id,
        "data_dict": {
            "X": X_df,
            "y": y_s,
        },
    }


def load_bnlearn_dataset(dataset_name: str) -> dict:
    """Load a dataset from bnlearn. (default to regression task)

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dict: Dictionary containing the features and target variables.
    """
    # ===== Load the dataset with id=====
    dataset_id = dataset2bnlearn_id[dataset_name]
    data_loaded = bn.import_example(data=dataset_id, n=10000)

    # ===== Get the features and target variables =====
    X_df = data_loaded
    # bnlearn may return the features in different orders
    X_df = X_df[X_df.columns.sort_values()]
    # Use a dummy target variable
    y_s = pd.Series(np.ones(X_df.shape[0]) * DUMMY_TARGET)

    return {
        "id": dataset_id,
        "data_dict": {
            "X": X_df,
            "y": y_s,
        },
    }


def load_pgmpy_dataset(dataset_name: str) -> dict:
    """Load a dataset from pgmpy. (default to regression task)

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        dict: Dictionary containing the features and target variables.
    """
    # ===== Load the dataset with id=====
    dataset_id = dataset2pgmpy_id[dataset_name]
    data_model = get_example_model(dataset_id)
    data_loaded = data_model.simulate(n_samples=10000, seed=42)

    # ===== Get the features and target variables =====
    X_df = data_loaded
    # pgmpy may return the features in different orders
    X_df = X_df[X_df.columns.sort_values()]
    # Use a dummy target variable
    y_s = pd.Series(np.ones(X_df.shape[0]) * DUMMY_TARGET)

    return {
        "id": dataset_id,
        "data_dict": {
            "X": X_df,
            "y": y_s,
        },
    }


def load_local_dataset(
    dataset_name: str,
    target_col: Optional[str] = None,
    metafeature_dict: Optional[dict] = None,
) -> dict:
    """Load a dataset from a local file.

    Args:
        dataset_name (str): Name of the dataset to load.
        target_col (Optional[str], optional): Name of the target column.
        metafeature_dict (Optional[dict], optional): Dictionary containing the metafeatures.

    Returns:
        dict: Dictionary containing the features and target variables.
    """
    # ===== Transform the dataset name to the path =====
    if dataset_name in dataset2path.keys():
        dataset_name = dataset2path[dataset_name]

    if "csv" in dataset_name:
        # Ensure the local samples are loaded with the predefined data types (if provided).
        # Pandas may infer wrong data types when loading CSV files, e.g., "1" -> "1.0".
        col2type = metafeature_dict.get("col2type", None)
        col2pandas_dtype = {col: col_type.pandas_dtype for col, col_type in col2type.items()} if col2type else None
        data_df = pd.read_csv(
            dataset_name,
            dtype=col2pandas_dtype,
        )
        # === Ignore the first column if it is an index ===
        if "Unnamed: 0" in data_df.columns:
            data_df = data_df.drop("Unnamed: 0", axis=1)
        # === Drop the last column as the target ===
        target_col = target_col if target_col in data_df.columns else data_df.columns[-1]
        X_df = data_df.drop(target_col, axis=1)
        y_s = data_df[target_col]
    elif "mat" in dataset_name:
        data = spio.loadmat(dataset_name)
        X_df = pd.DataFrame(data["X"])
        y_s = pd.Series(data["Y"][:, 0])
    else:
        raise NotImplementedError("File format not recognised for local dataset.")

    return {
        "id": None,
        "data_dict": {
            "X": X_df,
            "y": y_s,
        },
    }
