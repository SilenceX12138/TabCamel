Tutorials
=========

This section contains step-by-step tutorials to help you master TabCamel.

Available Tutorials
-------------------

The tutorials are provided as interactive Jupyter notebooks that you can run locally or in Google Colab.

Tutorial 1: TabularDataset Basics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <a href="https://colab.research.google.com/github/SilenceX12138/TabCamel/blob/master/docs/tutorial/tutorial1_tabular_dataset.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

**Topics covered:**
  * Loading datasets from various sources
  * Understanding dataset properties
  * Basic data exploration
  * Working with remote and local datasets

**What you'll learn:**
  * How to create TabularDataset instances
  * Accessing dataset properties and metadata
  * Loading data from OpenML, UCI, and local files
  * Understanding feature types and target variables

Tutorial 2: Data Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <a href="https://colab.research.google.com/github/SilenceX12138/TabCamel/blob/master/docs/tutorial/tutorial2_transform.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

**Topics covered:**
  * Using built-in transformations
  * Creating custom transformations
  * Fit-transform workflow
  * Handling different data types

**What you'll learn:**
  * Working with the BaseTransform class
  * Preprocessing numerical and categorical features
  * Building transformation pipelines
  * Best practices for data preprocessing

Tutorial 3: Advanced Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <a href="https://colab.research.google.com/github/SilenceX12138/TabCamel/blob/master/docs/tutorial/tutorial3_pipeline.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

**Topics covered:**
  * Combining sampling, splitting, and transformations
  * Building end-to-end ML workflows
  * Integration with popular ML libraries
  * Performance optimization

**What you'll learn:**
  * Creating complete ML pipelines
  * Combining TabCamel with scikit-learn and other libraries
  * Best practices for workflow design
  * Performance tips and tricks

Quick Start Tutorial
--------------------

If you want to get started immediately, here's a condensed tutorial covering the basics:

Step 1: Installation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install tabcamel

Step 2: Load Your First Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tabcamel.data.dataset import TabularDataset
   
   # Load the famous iris dataset
   dataset = TabularDataset('iris', task_type='classification')
   
   # Display basic information
   print(f"Dataset: {dataset.dataset_name}")
   print(f"Samples: {dataset.num_samples}")
   print(f"Features: {dataset.num_features}")
   print(f"Classes: {dataset.num_classes}")

Step 3: Explore the Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access the data
   print("First 5 rows:")
   print(dataset.data_df.head())
   
   # Check feature types
   print("\\nFeature types:")
   print(dataset.col2type)
   
   # Look at class distribution
   print("\\nClass distribution:")
   print(dataset.class2distribution)

Step 4: Sample and Split
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a stratified sample
   sample_result = dataset.sample('stratified', sample_size=100)
   sampled_dataset = sample_result['dataset_sampled']
   
   # Split into train/test
   split_result = sampled_dataset.split('stratified', train_size=0.8)
   train_set = split_result['train_set']
   test_set = split_result['test_set']
   
   print(f"Training samples: {train_set.num_samples}")
   print(f"Test samples: {test_set.num_samples}")

Step 5: Work with Local Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import tabcamel.utils.config as config
   
   # Configure local datasets
   config.set_local_data_path({
       "my_data": "/path/to/your/dataset.csv"
   })
   
   # Load local dataset
   local_dataset = TabularDataset('my_data', task_type='classification')

Tutorial Deep Dives
-------------------

Working with Different Data Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**OpenML Datasets:**

.. code-block:: python

   # Popular classification datasets
   datasets = ['iris', 'adult', 'titanic', 'wine']
   
   for name in datasets:
       dataset = TabularDataset(name, task_type='classification')
       print(f"{name}: {dataset.num_samples} samples, {dataset.num_features} features")

**UCI Repository:**

.. code-block:: python

   # UCI datasets
   uci_datasets = ['mushroom', 'abalone']
   
   for name in uci_datasets:
       dataset = TabularDataset(name, task_type='classification')
       print(f"{name}: {dataset.info_df}")

**Specialized Sources:**

.. code-block:: python

   # Bayesian network datasets
   bn_datasets = ['asia', 'cancer', 'alarm']
   
   for name in bn_datasets:
       dataset = TabularDataset(name, task_type='classification')
       print(f"{name}: {dataset.num_classes} classes")

Advanced Sampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Balanced vs Imbalanced Sampling:**

.. code-block:: python

   # Load an imbalanced dataset
   dataset = TabularDataset('adult', task_type='classification')
   print("Original distribution:", dataset.class2distribution)
   
   # Maintain original distribution
   stratified = dataset.sample('stratified', sample_size=1000)
   print("Stratified distribution:", stratified['dataset_sampled'].class2distribution)
   
   # Create balanced distribution
   uniform = dataset.sample('uniform', sample_size=1000)  # Equal per class
   print("Uniform distribution:", uniform['dataset_sampled'].class2distribution)

**Custom Distribution Sampling:**

.. code-block:: python

   # Create custom class balance
   custom_dist = {
       ' <=50K': 0.7,  # 70% negative class
       ' >50K': 0.3    # 30% positive class  
   }
   
   custom_sample = dataset.sample(
       'customised_ratio',
       sample_size=1000,
       customised_class2distribution=custom_dist
   )

Splitting Strategies for Different Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Time Series Aware Splitting:**

.. code-block:: python

   # For time series data, use fixed splits
   total_samples = dataset.num_samples
   split_point = int(0.8 * total_samples)
   
   train_indices = list(range(split_point))
   test_indices = list(range(split_point, total_samples))
   
   split_result = dataset.split(
       'fixed',
       indices_train=train_indices,
       indices_test=test_indices
   )

**Cross-Validation Setup:**

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold
   
   # Prepare for k-fold cross-validation
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
   folds = []
   for train_idx, test_idx in skf.split(dataset.X_df, dataset.y_df):
       fold = dataset.split(
           'fixed',
           indices_train=train_idx.tolist(),
           indices_test=test_idx.tolist()
       )
       folds.append(fold)

Working with Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Creating a Preprocessing Pipeline:**

.. code-block:: python

   from tabcamel.data.transform import BaseTransform
   from sklearn.preprocessing import StandardScaler, LabelEncoder
   
   class PreprocessingPipeline(BaseTransform):
       def __init__(self):
           super().__init__()
           self.scaler = StandardScaler()
           self.encoders = {}
           
       def _fit(self, data_df):
           # Fit scalers for numerical features
           numerical_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
           if len(numerical_cols) > 0:
               self.scaler.fit(data_df[numerical_cols])
           
           # Fit encoders for categorical features  
           categorical_cols = data_df.select_dtypes(include=['object']).columns
           for col in categorical_cols:
               encoder = LabelEncoder()
               encoder.fit(data_df[col].astype(str))
               self.encoders[col] = encoder
               
       def _transform(self, data_df):
           result_df = data_df.copy()
           
           # Scale numerical features
           numerical_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
           if len(numerical_cols) > 0:
               result_df[numerical_cols] = self.scaler.transform(data_df[numerical_cols])
           
           # Encode categorical features
           for col, encoder in self.encoders.items():
               if col in result_df.columns:
                   result_df[col] = encoder.transform(result_df[col].astype(str))
                   
           return result_df

Integration with ML Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**With scikit-learn:**

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report
   
   # Prepare data
   split_result = dataset.split('stratified', test_size=0.2)
   train_set = split_result['train_set']
   test_set = split_result['test_set']
   
   # Train model
   model = RandomForestClassifier(random_state=42)
   model.fit(train_set.X_df, train_set.y_df.values.ravel())
   
   # Evaluate
   predictions = model.predict(test_set.X_df)
   print(classification_report(test_set.y_df, predictions))

**With AutoGluon:**

.. code-block:: python

   try:
       from autogluon.tabular import TabularPredictor
       
       # Prepare data for AutoGluon
       train_data = train_set.data_df
       test_data = test_set.data_df
       
       # Train predictor
       predictor = TabularPredictor(
           label=dataset.target_col,
           path='./autogluon_models/'
       ).fit(train_data, time_limit=300)
       
       # Evaluate
       performance = predictor.evaluate(test_data)
       print(f"Test accuracy: {performance}")
       
   except ImportError:
       print("AutoGluon not installed. Install with: pip install autogluon")

Troubleshooting Common Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Memory Management:**

.. code-block:: python

   # For large datasets, sample first
   if dataset.num_samples > 50000:
       # Work with a sample for experimentation
       sample_result = dataset.sample('stratified', sample_size=10000)
       working_dataset = sample_result['dataset_sampled']
   else:
       working_dataset = dataset

**Handling Missing Data:**

.. code-block:: python

   # Check for missing data
   print("Missing values per column:")
   print(dataset.data_df.isnull().sum())
   
   # Handle missing values before transformation
   clean_df = dataset.data_df.fillna(method='ffill')  # Forward fill
   
   # Create new dataset with clean data
   clean_dataset = TabularDataset(
       dataset_name=dataset.dataset_name + '_clean',
       task_type=dataset.task_type,
       target_col=dataset.target_col,
       data_df=clean_df
   )

**Debugging Dataset Issues:**

.. code-block:: python

   # Inspect dataset thoroughly
   print("Dataset info:")
   print(dataset.info_df)
   
   print("\\nData types:")
   print(dataset.col2type)
   
   print("\\nFirst few rows:")
   print(dataset.data_df.head())
   
   print("\\nDataset statistics:")
   print(dataset.data_df.describe())

Next Steps
----------

After completing these tutorials, you should be able to:

1. Load and explore tabular datasets from various sources
2. Apply different sampling and splitting strategies
3. Create and use data transformations
4. Build complete ML workflows with TabCamel
5. Integrate TabCamel with other ML libraries
6. Handle common data processing challenges

For more advanced topics, check out:

* :doc:`user_guide` for comprehensive documentation
* :doc:`api_reference` for detailed API information  
* :doc:`examples` for real-world use cases
