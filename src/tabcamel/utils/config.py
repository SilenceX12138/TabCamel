"""
Configuration utilities for TabCamel
====================================

This module provides utilities for managing TabCamel configuration,
particularly for setting up and managing local dataset paths.
"""

from .. import LOCAL_DATA_PATH


def get_data_path() -> str:
    """Get the current data path for local datasets.

    Returns:
        str: The current local data path.
    """
    return LOCAL_DATA_PATH


def set_data_path(path: str) -> None:
    """Set the data path for local datasets.

    This function sets the TABCAMEL_DATA_PATH environment variable
    for the current session. To make it permanent, users should set
    this environment variable in their shell configuration.

    Args:
        path (str): The new path for local datasets.

    Note:
        This only affects the current Python session. To make the change
        permanent, set the TABCAMEL_DATA_PATH environment variable in your
        shell configuration (.bashrc, .zshrc, etc.).
    """
    # Update the module-level variable
    global LOCAL_DATA_PATH
    LOCAL_DATA_PATH = path


def get_dataset_info() -> dict:
    """Get information about available dataset categories and sources.

    Returns:
        dict: Information about available datasets and sources.
    """
    from .. import (dataset2bnlearn_id, dataset2openml_id, dataset2path, dataset2pgmpy_id, dataset2sklearn_id,
                    dataset2uci_id)

    return {
        "remote_sources": {
            "openml": {
                "count": len(dataset2openml_id),
                "description": "Datasets from OpenML repository",
                "examples": list(dataset2openml_id.keys())[:5],
            },
            "uci": {
                "count": len(dataset2uci_id),
                "description": "Datasets from UCI ML Repository",
                "examples": list(dataset2uci_id.keys())[:5],
            },
            "sklearn": {
                "count": len(dataset2sklearn_id),
                "description": "Built-in scikit-learn datasets",
                "examples": list(dataset2sklearn_id.keys()),
            },
            "bnlearn": {
                "count": len(dataset2bnlearn_id),
                "description": "Datasets from bnlearn R package",
                "examples": list(dataset2bnlearn_id.keys()),
            },
            "pgmpy": {
                "count": len(dataset2pgmpy_id),
                "description": "Bayesian network datasets from pgmpy",
                "examples": list(dataset2pgmpy_id.keys())[:5],
            },
        },
        "local_datasets": {
            "count": len(dataset2path),
            "description": "Predefined local datasets (require data files)",
            "examples": list(dataset2path.keys())[:5],
            "data_path": get_data_path(),
        },
        "usage_info": {
            "remote": "Use dataset name directly: TabularDataset('iris', task_type='classification')",
            "local_predefined": "Use predefined name: TabularDataset('lung', task_type='classification')",
            "local_custom": "Use full file path: TabularDataset('/path/to/your/data.csv', task_type='classification')",
        },
    }


def list_available_datasets() -> None:
    """Print a comprehensive list of all available datasets."""
    info = get_dataset_info()

    print("TabCamel Available Datasets")
    print("=" * 50)
    print(f"Local data path: {info['local_datasets']['data_path']}")
    print()

    # Remote datasets
    print("Remote Datasets (automatically downloaded):")
    print("-" * 40)
    for source, details in info["remote_sources"].items():
        print(f"{source.upper()}: {details['count']} datasets")
        print(f"  Description: {details['description']}")
        if details["examples"]:
            print(f"  Examples: {', '.join(details['examples'])}")
        print()

    # Local datasets
    print("Local Datasets (require data files):")
    print("-" * 40)
    print(f"Predefined: {info['local_datasets']['count']} datasets")
    print(f"  Description: {info['local_datasets']['description']}")
    if info["local_datasets"]["examples"]:
        print(f"  Examples: {', '.join(info['local_datasets']['examples'])}")
    print()

    # Usage examples
    print("Usage Examples:")
    print("-" * 40)
    for usage_type, example in info["usage_info"].items():
        print(f"  {usage_type}: {example}")
    print()

    print("Configuration:")
    print("-" * 40)
    print("  Set custom data path: tabcamel.utils.config.set_data_path('/your/path')")
    print("  Environment variable: export TABCAMEL_DATA_PATH='/your/path'")
