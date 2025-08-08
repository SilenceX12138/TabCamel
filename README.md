# TabCamel


[![Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SilenceX12138/TabCamel/blob/master/docs/tutorial/tutorial1_tabular_dataset.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://static.pepy.tech/badge/tabcamel)](https://pypi.org/project/tabcamel/)

A DataFrame-focused solution for tabular datasets in machine learning workflows.

## 🎯 Features

- **TabularDataset**: Comprehensive dataset class with sampling and splitting capabilities
- **Data Transformations**: Scikit-learn compatible preprocessing transformations
- **Multi-source Loading**: Support for local files and popular ML repositories
- **AutoGluon Integration**: Seamless integration with AutoGluon for automated ML

## 🛠 Installation

```bash
pip install tabcamel
```

## 🚀 Quick Start

```python
from tabcamel.data.dataset import TabularDataset

# Load a remote dataset
dataset = TabularDataset('iris', task_type='classification')

# Split into train/test sets
train_test = dataset.split('stratified', train_size=0.8)
train_data = train_test['train_set']
test_data = train_test['test_set']

print(train_data)
```

## 💽 Dataset Sources

TabCamel supports multiple data sources:

### Remote Datasets

- **OpenML**: 30+ popular datasets (`'iris'`, `'adult'`, `'titanic'`, etc.)
- **UCI ML Repository**: Classic datasets with proper metadata
- **scikit-learn**: Built-in sklearn datasets (`'diabetes'`, etc.)
- **pgmpy**: Bayesian network datasets from [pgmpy](https://github.com/pgmpy/pgmpy)
- **bnlearn**: datasets from [bnlearn](https://erdogant.github.io/bnlearn/pages/html/index.html)

### Local Datasets

For local datasets, you have several options:

#### Option 1: Direct File Path

```python
# Use full path to your dataset
dataset = TabularDataset('/path/to/your/data.csv', task_type='classification')
```

#### Option 2: Configure Data Directory

```python
import tabcamel.utils.config as config

# Set up your data directory
local_dataset2path = {
    "local_data": "/path/to/your/data.csv",
}
config.set_local_data_path(local_dataset2path)

# Now use short names
dataset = TabularDataset('local_data', task_type='classification')
```

## 💻 Examples

### Basic Usage

```python
from tabcamel.data.dataset import TabularDataset

# Remote dataset
dataset = TabularDataset('adult', task_type='classification')

# Local dataset with full path
dataset = TabularDataset('/home/user/data/my_data.csv', task_type='regression')

# Local dataset with configured data directory
dataset = TabularDataset('my_data', task_type='classification')
```

### Data Operations

```python
# Dataset sampling
sample_result = dataset.sample('stratified', sample_size=1000)
sampled_data = sample_result['dataset_sampled']

# Dataset splitting
split_result = dataset.split('stratified', test_size=0.2)
train_set = split_result['train_set']
test_set = split_result['test_set']

# Access properties
print(f"Samples: {dataset.num_samples}")
print(f"Features: {dataset.num_features}")
print(f"Classes: {dataset.num_classes}")
```


## 📚 Citation

If you use TabCamel in your research, please cite:

```bibtex
@misc{tabcamel,
  title = {TabCamel: A DataFrame-focused solution for tabular datasets in machine learning workflows},
  author = {Xiangjian Jiang},
  year = {2025},
  howpublished = {\url{https://github.com/SilenceX12138/TabCamel}},
}
```