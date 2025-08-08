User Guide
==========

This comprehensive guide covers all aspects of using TabCamel for tabular data processing.

TabularDataset Class
--------------------

The ``TabularDataset`` class is the core component of TabCamel, providing a comprehensive interface for handling tabular datasets in machine learning workflows.

Basic Initialization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tabcamel.data.dataset import TabularDataset
   
   # Basic initialization
   dataset = TabularDataset(
       dataset_name='iris',
       task_type='classification',
       target_col=None,  # Optional: specify target column
       metafeature_dict=None,  # Optional: custom metadata
       data_df=None,  # Optional: provide DataFrame directly
   )

Parameters Explained
~~~~~~~~~~~~~~~~~~~~

**dataset_name**: The identifier for your dataset. Can be:
  * A predefined dataset name (e.g., 'iris', 'adult', 'titanic')
  * A local file path (e.g., '/path/to/data.csv')
  * A configured short name (after using ``config.set_local_data_path()``)

**task_type**: The machine learning task type:
  * ``'classification'``: For categorical target variables
  * ``'regression'``: For continuous target variables

**target_col**: (Optional) Name of the target column:
  * If ``None``, the last column is used as target and renamed to 'target'
  * Can specify custom column name

Dataset Properties
~~~~~~~~~~~~~~~~~~

The ``TabularDataset`` class provides many useful properties:

**Basic Properties:**

.. code-block:: python

   print(f"Dataset name: {dataset.dataset_name}")
   print(f"Task type: {dataset.task_type}")
   print(f"Number of samples: {dataset.num_samples}")
   print(f"Number of features: {dataset.num_features}")
   print(f"Target column: {dataset.target_col}")

**Classification-specific Properties:**

.. code-block:: python

   if dataset.task_type == 'classification':
       print(f"Number of classes: {dataset.num_classes}")
       print(f"Class list: {dataset.class_list}")
       print(f"Class distribution: {dataset.class2distribution}")
       print(f"Samples per class: {dataset.class2samples}")

**Feature Type Information:**

.. code-block:: python

   print(f"Column types: {dataset.col2type}")
   print(f"Numerical features: {dataset.numerical_feature_list}")
   print(f"Categorical features: {dataset.categorical_feature_list}")

Data Access
~~~~~~~~~~~

Access the underlying data:

.. code-block:: python

   # Full DataFrame (features + target)
   full_data = dataset.data_df
   
   # Features only
   features = dataset.X_df
   
   # Target only  
   target = dataset.y_df
   
   # Data indices
   indices = dataset.data_indices

Sampling Methods
----------------

TabCamel provides several sampling strategies to create subsets of your data.

Random Sampling
~~~~~~~~~~~~~~~

Random sampling selects samples uniformly at random:

.. code-block:: python

   # Random sampling with fixed size
   result = dataset.sample('random', sample_size=1000)
   sampled_dataset = result['dataset_sampled']
   sample_indices = result['sample_indices']

Stratified Sampling
~~~~~~~~~~~~~~~~~~~

Stratified sampling maintains the class distribution of the original dataset:

.. code-block:: python

   # Stratified sampling (maintains original class balance)
   result = dataset.sample('stratified', sample_size=1000)
   sampled_dataset = result['dataset_sampled']
   
   # Check that class distribution is maintained
   print("Original distribution:", dataset.class2distribution)
   print("Sampled distribution:", sampled_dataset.class2distribution)

Uniform Sampling
~~~~~~~~~~~~~~~~

Uniform sampling ensures equal representation from each class:

.. code-block:: python

   # Uniform sampling (equal samples per class)
   # Note: sample_size must be divisible by number of classes
   result = dataset.sample('uniform', sample_size=150)  # 50 per class for 3-class dataset

Custom Ratio Sampling
~~~~~~~~~~~~~~~~~~~~~

Specify custom class distributions:

.. code-block:: python

   # Custom class distribution
   custom_distribution = {
       'class_A': 0.5,  # 50%
       'class_B': 0.3,  # 30% 
       'class_C': 0.2,  # 20%
   }
   
   result = dataset.sample(
       'customised_ratio', 
       sample_size=1000,
       customised_class2distribution=custom_distribution
   )

Fixed Sampling
~~~~~~~~~~~~~~

Sample specific indices:

.. code-block:: python

   # Sample specific rows by index
   specific_indices = [0, 10, 20, 30, 40]
   result = dataset.sample('fixed', sample_indices=specific_indices)

Data Splitting
--------------

Split your dataset into training and testing sets with various strategies.

Random Split
~~~~~~~~~~~~

.. code-block:: python

   # Random 80-20 split
   split_result = dataset.split('random', train_size=0.8)
   train_set = split_result['train_set']
   test_set = split_result['test_set']
   train_indices = split_result['indices_train']
   test_indices = split_result['indices_test']

Stratified Split
~~~~~~~~~~~~~~~~

Maintains class balance in both splits:

.. code-block:: python

   # Stratified split ensuring class balance
   split_result = dataset.split('stratified', test_size=0.2)
   
   print("Original distribution:", dataset.class2distribution)
   print("Train distribution:", split_result['train_set'].class2distribution)
   print("Test distribution:", split_result['test_set'].class2distribution)

Fixed Split
~~~~~~~~~~~

Use predetermined indices:

.. code-block:: python

   # Pre-defined train/test indices
   train_indices = [0, 1, 2, 3, 4]
   test_indices = [5, 6, 7, 8, 9]
   
   split_result = dataset.split(
       'fixed',
       indices_train=train_indices,
       indices_test=test_indices
   )

Data Transformations
--------------------

TabCamel provides scikit-learn compatible transformations for data preprocessing.

Base Transform Class
~~~~~~~~~~~~~~~~~~~~

All transformations inherit from ``BaseTransform``:

.. code-block:: python

   from tabcamel.data.transform import BaseTransform
   
   # Example custom transform
   class CustomTransform(BaseTransform):
       def _fit(self, data_df):
           # Learn parameters from data
           pass
           
       def _transform(self, data_df):
           # Apply transformation
           return transformed_data
           
       def _inverse_transform(self, data_df):
           # Reverse transformation
           return original_data

Using Transforms
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create and fit a transform
   transform = SomeTransform()
   transform.fit(train_data.X_df)
   
   # Transform data
   transformed_train = transform.transform(train_data.X_df)
   transformed_test = transform.transform(test_data.X_df)
   
   # Inverse transform (if supported)
   original_data = transform.inverse_transform(transformed_train)

Available Data Sources
----------------------

OpenML Datasets
~~~~~~~~~~~~~~~

TabCamel provides access to 30+ popular OpenML datasets:

**Classification datasets:**
  * 'iris': Classic iris flower dataset
  * 'adult': Adult income prediction
  * 'titanic': Titanic passenger survival
  * 'wine': Wine quality classification
  * And many more...

**Regression datasets:**
  * 'california_housing': California housing prices
  * 'diabetes': Diabetes progression
  * 'house_16H': House prices

UCI ML Repository
~~~~~~~~~~~~~~~~~

Access UCI datasets:

.. code-block:: python

   # UCI datasets
   dataset = TabularDataset('mushroom', task_type='classification')
   dataset = TabularDataset('abalone', task_type='classification') 

Scikit-learn Datasets
~~~~~~~~~~~~~~~~~~~~~

Built-in sklearn datasets:

.. code-block:: python

   dataset = TabularDataset('diabetes', task_type='regression')

Specialized Sources
~~~~~~~~~~~~~~~~~~~

**Bayesian Network Datasets (pgmpy):**

.. code-block:: python

   # Small networks
   dataset = TabularDataset('asia', task_type='classification')
   dataset = TabularDataset('cancer', task_type='classification')
   
   # Medium networks  
   dataset = TabularDataset('alarm', task_type='classification')
   
   # Large networks
   dataset = TabularDataset('hailfinder', task_type='classification')

**bnlearn Datasets:**

.. code-block:: python

   dataset = TabularDataset('auto_mpg', task_type='regression')

Configuration Management
------------------------

Configure TabCamel for your specific needs.

Setting Data Paths
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tabcamel.utils.config as config
   
   # Set local dataset paths
   local_datasets = {
       "my_classification_data": "/path/to/classification.csv",
       "my_regression_data": "/path/to/regression.csv",
       "project_data": "/project/data/dataset.xlsx",
   }
   config.set_local_data_path(local_datasets)
   
   # Now use short names
   dataset = TabularDataset('my_classification_data', task_type='classification')

Viewing Available Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # List all configured datasets
   config.list_available_datasets()
   
   # View dataset mapping
   print(config.get_dataset_mapping())

Advanced Features
-----------------

Dataset Information
~~~~~~~~~~~~~~~~~~~

Get comprehensive dataset statistics:

.. code-block:: python

   # Detailed information DataFrame
   info_df = dataset.info_df
   print(info_df)
   
   # String representation
   print(str(dataset))
   print(repr(dataset))

Working with Metafeatures
~~~~~~~~~~~~~~~~~~~~~~~~~

Store and access custom metadata:

.. code-block:: python

   # Create dataset with custom metadata
   custom_metadata = {
       "source": "experiment_2024",
       "preprocessing": "standard_scaling",
       "notes": "High quality dataset"
   }
   
   dataset = TabularDataset(
       'my_data',
       task_type='classification',
       metafeature_dict=custom_metadata
   )
   
   # Access metadata
   print(dataset.metafeature_dict)

Data Quality Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Remove classes with too few samples
   dataset.drop_low_sample_class(min_sample_per_class=10)
   
   # Remove specific class
   dataset.drop_class('unwanted_class')

Error Handling and Best Practices
----------------------------------

Common Issues
~~~~~~~~~~~~~

**Target Column Issues:**

.. code-block:: python

   # Always specify task_type correctly
   dataset = TabularDataset('data.csv', task_type='classification')  # Not 'clf'
   
   # Handle missing target column gracefully
   try:
       dataset = TabularDataset('data.csv', task_type='classification', target_col='label')
   except KeyError:
       print("Target column 'label' not found in dataset")

**Sampling Issues:**

.. code-block:: python

   # Ensure sample size is reasonable
   max_samples = dataset.num_samples
   sample_size = min(1000, max_samples)  # Don't exceed dataset size
   
   # For uniform sampling, ensure divisibility
   if dataset.num_classes and sample_size % dataset.num_classes != 0:
       sample_size = (sample_size // dataset.num_classes) * dataset.num_classes

**Split Issues:**

.. code-block:: python

   # Ensure splits include all classes for classification
   try:
       split_result = dataset.split('stratified', test_size=0.1)
   except ValueError as e:
       print(f"Stratified split failed: {e}")
       # Use random split as fallback
       split_result = dataset.split('random', test_size=0.1)

Best Practices
~~~~~~~~~~~~~~

1. **Always specify task_type explicitly**
2. **Use stratified sampling/splitting for classification tasks**
3. **Check dataset properties before operations**
4. **Handle edge cases (small datasets, imbalanced classes)**
5. **Use configuration for frequently accessed datasets**
6. **Validate data quality after loading**

Performance Tips
~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, sample first then split
   if dataset.num_samples > 100000:
       sampled = dataset.sample('stratified', sample_size=10000)
       split_result = sampled['dataset_sampled'].split('stratified', test_size=0.2)
   else:
       split_result = dataset.split('stratified', test_size=0.2)
   
   # Reuse datasets when possible
   train_set = split_result['train_set'] 
   # train_set is also a TabularDataset with all methods available
