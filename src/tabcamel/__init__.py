"""
Camel: A DataFrame-focused Solution for Tabular Datasets
=======================================================

Camel is a comprehensive Python library for handling tabular datasets in machine learning
workflows. It provides a unified interface for data loading, preprocessing, transformation,
and augmentation with built-in support for various data sources and formats.

Key Features:
    - **TabularDataset**: Comprehensive dataset class with sampling and splitting capabilities
    - **Data Transformations**: Scikit-learn compatible preprocessing transformations
    - **Multi-source Loading**: Support for local files, remote datasets, and popular ML repositories
    - **AutoGluon Integration**: Seamless integration with AutoGluon for automated ML

Main Components:
    - :mod:`camel.data`: Core data handling classes and transformations
    - :mod:`camel.utils`: Utility functions for data loading and processing

Example:
    >>> from tabcamel.data.dataset import TabularDataset
    >>> dataset = TabularDataset('iris', task_type='classification')
    >>> train_test = dataset.split('stratified', train_size=0.8)
    >>> train_data = train_test['train_set']
"""

import os
import warnings


# ================================================================
# =                                                              =
# =                    Warning Information                       =
# =                                                              =
# ================================================================
def color_formatwarning(message, category, filename, lineno, line=""):
    COLOR = "\033[33m"  # Yellow
    RESET = "\033[0m"
    BOLD = "\033[1m"

    return f"{COLOR}{BOLD}{category.__name__}: {filename}:{lineno}: {RESET}{COLOR}{message}{RESET}\n"


warnings.formatwarning = color_formatwarning

# ================================================================
# =                                                              =
# =                    Dataset Information                       =
# =                                                              =
# ================================================================
DUMMY_TARGET = 12138
LOCAL_DATA_PATH = "/home/xj265/phd/codebase/Camel/dataset/"

dataset2openml_id = {
    # Classification
    "qsar-biodeg"               : 1494,
    "vehicle"                   : 54,
    "texture"                   : 40499,
    "steel-plates-fault"        : 1504,
    "MiceProtein"               : 40966,
    "mfeat-fourier"             : 14,
    "adult"                     : 179,
    "one-hundred-plants-texture": 1493,
    "energy-efficiency"         : 1472,
    "collins"                   : 40971,
    "soybean"                   : 42,
    "wilt"                      : 40983,
    "autoUniv-au6-1000"         : 1555,
    "vowel"                     : 307,
    "stock"                     : 841,
    "iris"                      : 61,
    "electricity"               : 151,
    "higgs"                     : 4532,
    "splice"                    : 46,
    "SpeedDating"               : 40536,
    "ada-agnostic"              : 1043,
    "artificial-characters"     : 1459,
    "jasmine"                   : 41143,
    "phoneme"                   : 1489,
    "credit-g"                  : 46378,
    "nomao"                     : 45078,
    "mfeat-zernike"             : 22,
    # Regression
    "liver-disorders"                : 8,
    "california_housing"             : 43939,
    "house_16H"                      : 574,
    "space_ga"                       : 507,
    "Sberbank_Russian_Housing_Market": 46787,
    "Ailerons"                       : 296,
    "elevators"                      : 216,
    "california_housing"             : 43939,
    "house_16H"                      : 574,
    "house_sales"                    : 42092,
}

dataset2uci_id = {
    # Classification
    "predict_students_dropout_and_academic_success": 697,
    "aids_clinical_trials_group_study_175"         : 890,
    "support2"                                     : 880,
    "mushroom"                                     : 73,
    "auction_verification"                         : 713,
    "abalone"                                      : 1,
    "statlog_german_credit_data"                   : 144,
    # Regression
    "Superconductivty": 464,
    "Wine Quality"    : 186,
}

dataset2sklearn_id = {
    # Regression
    "diabetes": "load_diabetes",
}

dataset2bnlearn_id = {
    # Continuous
    "auto_mpg": "auto_mpg",
}

dataset2pgmpy_id = {
    # Small Networks (< 20 nodes)
    "asia"      : "asia",
    "cancer"    : "cancer",
    "earthquake": "earthquake",
    # Medium Networks (20-50 nodes)
    "alarm" : "alarm",
    "barley": "barley",
    "water" : "water",
    # Large Networks (50-100 nodes)
    "hailfinder": "hailfinder",
    "hepar2"    : "hepar2",
    "win95pts"  : "win95pts",
    # Very Large Networks (100-1000 nodes)
    "andes"      : "andes",
    "diabetes_bn": "diabetes",
    "link"       : "link",
}

dataset2path = {
    # === Normal ===
    # Classification
    "lung": os.path.join(LOCAL_DATA_PATH, "lung/lung.mat"),
    # === Structure Learning (dataset__label) ===
    # Classification
    "cancer__Cancer"       : os.path.join(LOCAL_DATA_PATH, "structure/cancer/cancer.csv"),
    "insurance__PropCost"  : os.path.join(LOCAL_DATA_PATH, "structure/insurance/insurance.csv"),
    "hailfinder__R5Fcst"   : os.path.join(LOCAL_DATA_PATH, "structure/hailfinder/hailfinder.csv"),
    "ANDES__GOAL_147"      : os.path.join(LOCAL_DATA_PATH, "structure/andes/andes.csv"),
    "sangiovese__Treatment": os.path.join(LOCAL_DATA_PATH, "structure/sangiovese/sangiovese.csv"),
    # Regression
    "MAGIC-IRRI__YLD": os.path.join(LOCAL_DATA_PATH, "structure/magic-irri/magic-irri.csv"),
    "ARTH150__570"   : os.path.join(LOCAL_DATA_PATH, "structure/arth150/arth150.csv"),
    "healthcare__O"  : os.path.join(LOCAL_DATA_PATH, "structure/healthcare/healthcare.csv"),
    "MEHRA__blh"     : os.path.join(LOCAL_DATA_PATH, "structure/mehra/mehra.csv"),
}

dataset_with_dag = [
    "cancer",
    "insurance",
    "hailfinder",
    "ANDES",
    "sangiovese",
    "MAGIC-IRRI",
    "ARTH150",
    "healthcare",
    "MEHRA",
]
