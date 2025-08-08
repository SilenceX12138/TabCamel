API Reference
=============

This section contains the complete API documentation for TabCamel.

Core Classes
------------

tabcamel.data.dataset
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tabcamel.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

TabularDataset
^^^^^^^^^^^^^^

.. autoclass:: tabcamel.data.dataset.TabularDataset
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: sample
   .. automethod:: split
   .. automethod:: drop_low_sample_class  
   .. automethod:: drop_class

   **Properties:**

   .. autoproperty:: dataset_name
   .. autoproperty:: task_type
   .. autoproperty:: target_col
   .. autoproperty:: data_df
   .. autoproperty:: X_df
   .. autoproperty:: y_df
   .. autoproperty:: data_indices
   .. autoproperty:: num_samples
   .. autoproperty:: num_features
   .. autoproperty:: num_classes
   .. autoproperty:: class_list
   .. autoproperty:: class2samples
   .. autoproperty:: class2distribution
   .. autoproperty:: col2type
   .. autoproperty:: metafeature_dict
   .. autoproperty:: numerical_feature_list
   .. autoproperty:: categorical_feature_list
   .. autoproperty:: is_tensor
   .. autoproperty:: info_df

Data Transformations
--------------------

tabcamel.data.transform
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tabcamel.data.transform
   :members:
   :undoc-members:
   :show-inheritance:

BaseTransform
^^^^^^^^^^^^^

.. autoclass:: tabcamel.data.transform.BaseTransform
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: transform
   .. automethod:: inverse_transform

   **Properties:**

   .. autoproperty:: is_fitted

   **Abstract Methods:**

   Subclasses must implement these methods:

   .. automethod:: _fit
      :abstract:

   .. automethod:: _transform
      :abstract:

   .. automethod:: _inverse_transform
      :abstract:

Utility Functions
-----------------

tabcamel.utils.loading
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tabcamel.utils.loading
   :members:
   :undoc-members:
   :show-inheritance:

tabcamel.utils.config
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tabcamel.utils.config
   :members:
   :undoc-members:
   :show-inheritance:

tabcamel.utils.dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: tabcamel.utils.dataframe
   :members:
   :undoc-members:
   :show-inheritance:

Data Types
----------

tabcamel.data.stype
~~~~~~~~~~~~~~~~~~~~

.. automodule:: tabcamel.data.stype
   :members:
   :undoc-members:
   :show-inheritance:

Constants and Mappings
----------------------

Dataset Mappings
~~~~~~~~~~~~~~~~

The following constants define mappings from dataset names to their respective source IDs:

**OpenML Datasets**

.. code-block:: python

   from tabcamel import dataset2openml_id
   
   # Classification datasets
   dataset2openml_id = {
       "iris": 61,
       "adult": 179,
       "titanic": 40945,
       "wine": 187,
       # ... and many more
   }

**UCI ML Repository Datasets**

.. code-block:: python

   from tabcamel import dataset2uci_id
   
   dataset2uci_id = {
       "mushroom": 73,
       "abalone": 1,
       "statlog_german_credit_data": 144,
       # ... and more
   }

**Scikit-learn Datasets**

.. code-block:: python

   from tabcamel import dataset2sklearn_id
   
   dataset2sklearn_id = {
       "diabetes": "load_diabetes",
       # ... and more
   }

**Specialized Datasets**

.. code-block:: python

   from tabcamel import dataset2pgmpy_id, dataset2bnlearn_id
   
   # Bayesian network datasets from pgmpy
   dataset2pgmpy_id = {
       "asia": "asia",
       "cancer": "cancer", 
       "alarm": "alarm",
       # ... and more
   }
   
   # Datasets from bnlearn
   dataset2bnlearn_id = {
       "auto_mpg": "auto_mpg",
       # ... and more
   }

Method Details
--------------

TabularDataset Methods
~~~~~~~~~~~~~~~~~~~~~~

**__init__(dataset_name, task_type, target_col=None, metafeature_dict=None, data_df=None)**

Initialize a TabularDataset instance.

:param dataset_name: Name or path of the dataset
:type dataset_name: str
:param task_type: Type of ML task ('classification' or 'regression')
:type task_type: str
:param target_col: Name of target column (optional)
:type target_col: str or None
:param metafeature_dict: Custom metadata dictionary (optional) 
:type metafeature_dict: dict or None
:param data_df: Pre-loaded DataFrame (optional)
:type data_df: pandas.DataFrame or None

:raises ValueError: If task_type is not 'classification' or 'regression'
:raises FileNotFoundError: If dataset_name points to non-existent file

**sample(sample_mode, sample_size=None, sample_indices=None, customised_class2distribution=None, random_state=42)**

Sample a subset of the dataset.

:param sample_mode: Sampling strategy ('random', 'stratified', 'uniform', 'customised_ratio', 'fixed')
:type sample_mode: str
:param sample_size: Number of samples to draw
:type sample_size: int or float or None
:param sample_indices: Specific indices to sample (for 'fixed' mode)
:type sample_indices: list[int] or None
:param customised_class2distribution: Custom class distribution (for 'customised_ratio' mode)
:type customised_class2distribution: dict or None
:param random_state: Random seed for reproducibility
:type random_state: int

:returns: Dictionary with 'dataset_sampled' and 'sample_indices' keys
:rtype: dict

**split(split_mode, train_size=None, test_size=None, indices_train=None, indices_test=None, random_state=42)**

Split dataset into training and test sets.

:param split_mode: Split strategy ('random', 'stratified', 'fixed')
:type split_mode: str
:param train_size: Size of training set (float for proportion, int for absolute)
:type train_size: float or int or None
:param test_size: Size of test set (float for proportion, int for absolute)
:type test_size: float or int or None
:param indices_train: Training indices (for 'fixed' mode)
:type indices_train: list[int] or None
:param indices_test: Test indices (for 'fixed' mode) 
:type indices_test: list[int] or None
:param random_state: Random seed for reproducibility
:type random_state: int

:returns: Dictionary with 'train_set', 'test_set', 'indices_train', and 'indices_test' keys
:rtype: dict

Transform Methods
~~~~~~~~~~~~~~~~~

**fit(data_df)**

Fit the transformer to training data.

:param data_df: Training data to fit on
:type data_df: pandas.DataFrame

**transform(data_df)**

Apply fitted transformation to data.

:param data_df: Data to transform
:type data_df: pandas.DataFrame

:returns: Transformed data
:rtype: pandas.DataFrame

:raises ValueError: If transformer not fitted

**inverse_transform(data_df)**

Reverse the transformation (if supported).

:param data_df: Transformed data to reverse
:type data_df: pandas.DataFrame

:returns: Original data
:rtype: pandas.DataFrame

:raises NotImplementedError: If inverse transform not supported

Exception Classes
-----------------

TabCamel uses standard Python exceptions:

* **ValueError**: For invalid parameter values or incompatible operations
* **FileNotFoundError**: For missing dataset files
* **KeyError**: For missing columns or invalid keys  
* **NotImplementedError**: For unimplemented functionality

Usage Examples
--------------

Basic Dataset Creation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tabcamel.data.dataset import TabularDataset
   
   # Remote dataset
   dataset = TabularDataset('iris', task_type='classification')
   
   # Local dataset
   dataset = TabularDataset('/path/to/data.csv', task_type='regression')
   
   # With custom target column
   dataset = TabularDataset('data.csv', task_type='classification', target_col='label')

Sampling Examples
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Random sampling
   result = dataset.sample('random', sample_size=100)
   
   # Stratified sampling  
   result = dataset.sample('stratified', sample_size=100)
   
   # Custom distribution
   custom_dist = {'class_A': 0.6, 'class_B': 0.4}
   result = dataset.sample('customised_ratio', sample_size=100, 
                          customised_class2distribution=custom_dist)

Splitting Examples
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Random split
   result = dataset.split('random', train_size=0.8)
   
   # Stratified split
   result = dataset.split('stratified', test_size=0.2)
   
   # Fixed split
   train_idx = [0, 1, 2, 3, 4]
   test_idx = [5, 6, 7, 8, 9] 
   result = dataset.split('fixed', indices_train=train_idx, indices_test=test_idx)

Transform Examples
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tabcamel.data.transform import BaseTransform
   
   # Custom transformer
   class MyTransform(BaseTransform):
       def _fit(self, data_df):
           # Learn parameters
           pass
           
       def _transform(self, data_df):
           # Apply transformation
           return transformed_df
           
       def _inverse_transform(self, data_df):
           # Reverse if possible
           return original_df
   
   # Usage
   transform = MyTransform()
   transform.fit(train_data)
   transformed = transform.transform(test_data)
