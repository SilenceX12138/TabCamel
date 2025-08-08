TabCamel Documentation
======================

Welcome to TabCamel, a DataFrame-focused solution for tabular datasets in machine learning workflows.

Overview
--------

TabCamel is a comprehensive Python library designed to streamline the handling of tabular datasets in machine learning workflows. It provides a unified interface for data loading, preprocessing, transformation, and augmentation with built-in support for various data sources and formats.

Key Features
------------

* **TabularDataset**: Comprehensive dataset class with sampling and splitting capabilities
* **Data Transformations**: Scikit-learn compatible preprocessing transformations  
* **Multi-source Loading**: Support for local files, remote datasets, and popular ML repositories
* **AutoGluon Integration**: Seamless integration with AutoGluon for automated ML
* **Flexible Configuration**: Easy configuration management for data paths and settings

Quick Start
-----------

Install TabCamel using pip:

.. code-block:: bash

   pip install tabcamel

Load and work with a dataset:

.. code-block:: python

   from tabcamel.data.dataset import TabularDataset
   
   # Load a remote dataset
   dataset = TabularDataset('iris', task_type='classification')
   
   # Split into train/test sets
   train_test = dataset.split('stratified', train_size=0.8)
   train_data = train_test['train_set']
   test_data = train_test['test_set']
   
   print(train_data)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   user_guide
   tutorials
   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
