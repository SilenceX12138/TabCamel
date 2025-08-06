# TabCamel

A DataFrame-focused solution for tabular datasets in machine learning workflows.

## Features

- **TabularDataset**: Comprehensive dataset class with sampling and splitting capabilities
- **Data Transformations**: Scikit-learn compatible preprocessing transformations
- **Multi-source Loading**: Support for local files, remote datasets, and popular ML repositories
- **AutoGluon Integration**: Seamless integration with AutoGluon for automated ML

## Installation

```bash
pip install tabcamel
```

## Quick Start

```python
from tabcamel.data.dataset import TabularDataset

# Load a remote dataset (automatically downloaded)
dataset = TabularDataset('iris', task_type='classification')

# Split into train/test sets
train_test = dataset.split('stratified', train_size=0.8)
train_data = train_test['train_set']
test_data = train_test['test_set']

print(train_data)
```

## Dataset Sources

TabCamel supports multiple data sources:

### Remote Datasets (Automatic Download)

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
config.set_data_path('/your/data/directory')

# Copy your datasets to the directory
config.copy_example_data('/path/to/your/file.csv', 'my_dataset')

# Now use short names
dataset = TabularDataset('my_dataset.csv', task_type='classification')
```

## Configuration

### Setting Up Local Data Directory

```python
import tabcamel.utils.config as config

# Set up your data directory
config.setup_data_directory('/your/data/directory')
TabCamel uses `~/.tabcamel/datasets/` as the default directory for local datasets. You can customize this:
```

```python
import tabcamel.utils.config as config

# View current configuration
config.get_data_path()

# Set custom data path for current session
config.set_data_path('/your/custom/path')

# Set up data directory (creates if missing)
config.setup_data_directory('/your/custom/path')

# View all available datasets
config.list_available_datasets()
```

## Examples

### Basic Usage

```python
from tabcamel.data.dataset import TabularDataset

# Remote dataset
dataset = TabularDataset('adult', task_type='classification')

# Local dataset with full path
dataset = TabularDataset('/home/user/data/my_data.csv', task_type='regression')

# Local dataset with configured data directory
dataset = TabularDataset('my_data.csv', task_type='classification')
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

## Development

### Setting up Development Environment

```bash
git clone https://github.com/your-username/TabCamel.git
cd TabCamel
pip install -e .
```
