Getting Started
===============

This guide will help you get up and running with TabCamel quickly.

Installation
------------

TabCamel requires Python 3.10 or higher. Install using pip:

.. code-block:: bash

   pip install tabcamel

Dependencies
~~~~~~~~~~~~

TabCamel includes the following key dependencies:

* **pandas**: DataFrame operations
* **numpy**: Numerical computations  
* **scikit-learn**: Machine learning utilities
* **bnlearn**: Bayesian network datasets
* **ucimlrepo**: UCI ML Repository access
* **pgmpy**: Probabilistic graphical models

Basic Usage
-----------

Creating Your First Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tabcamel.data.dataset import TabularDataset
   
   # Load a popular classification dataset
   dataset = TabularDataset('iris', task_type='classification')
   
   # Display basic information
   print(dataset)
   print(f"Samples: {dataset.num_samples}")
   print(f"Features: {dataset.num_features}")
   print(f"Classes: {dataset.num_classes}")

Working with Local Data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Option 1: Direct file path
   dataset = TabularDataset('/path/to/your/data.csv', task_type='classification')
   
   # Option 2: Configure data directory
   import tabcamel.utils.config as config
   
   local_dataset2path = {
       "my_data": "/path/to/your/data.csv",
   }
   config.set_local_data_path(local_dataset2path)
   
   # Now use short names
   dataset = TabularDataset('my_data', task_type='classification')

Data Operations
~~~~~~~~~~~~~~~

**Sampling:**

.. code-block:: python

   # Random sampling
   sample_result = dataset.sample('random', sample_size=1000)
   sampled_data = sample_result['dataset_sampled']
   
   # Stratified sampling (maintains class balance)
   sample_result = dataset.sample('stratified', sample_size=1000)

**Splitting:**

.. code-block:: python

   # Random split
   split_result = dataset.split('random', train_size=0.8)
   train_set = split_result['train_set']
   test_set = split_result['test_set']
   
   # Stratified split (maintains class balance)
   split_result = dataset.split('stratified', test_size=0.2)

Supported Data Sources
----------------------

Remote Datasets
~~~~~~~~~~~~~~~

TabCamel supports multiple remote data sources:

**OpenML** (30+ datasets):
  * Popular datasets like 'iris', 'adult', 'titanic', 'wine'
  * Automatic metadata parsing
  * Consistent data format

**UCI ML Repository**:
  * Classic machine learning datasets
  * Proper metadata handling
  * Quality assured datasets

**scikit-learn**:
  * Built-in sklearn datasets like 'diabetes'
  * Integrated with sklearn ecosystem

**Specialized Sources**:
  * **pgmpy**: Bayesian network datasets ('asia', 'alarm', 'cancer')
  * **bnlearn**: Additional Bayesian datasets ('auto_mpg')

Local Datasets
~~~~~~~~~~~~~~

TabCamel can work with local files:

* **CSV files**: Most common format
* **Excel files**: .xlsx support
* **Custom formats**: Extensible loading system

Configuration
~~~~~~~~~~~~~

Set up custom data paths:

.. code-block:: python

   import tabcamel.utils.config as config
   
   # Configure local dataset paths
   config.set_local_data_path({
       "dataset1": "/path/to/dataset1.csv",
       "dataset2": "/path/to/dataset2.xlsx",
   })
   
   # List available datasets
   config.list_available_datasets()

Next Steps
----------

* Read the :doc:`user_guide` for detailed information about TabCamel's capabilities
* Check out the :doc:`tutorials` for hands-on examples
* Browse the :doc:`api_reference` for complete API documentation
* Explore the :doc:`examples` for real-world use cases
