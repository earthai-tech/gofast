gofast.tools.baseutils
======================

Overview
--------
The ``gofast.tools.baseutils`` module is a comprehensive collection of essential utilities and 
helper functions designed to facilitate data preprocessing, feature engineering, and the 
application of text vectorization techniques in data science and machine learning projects.

Key Features
------------
- **audit_data**: Conducts a thorough audit of the DataFrame, applying various preprocessing techniques to enhance data quality for analysis or modeling.
- **read_data**: Reads data from various formats into a pandas DataFrame, simplifying the initial data loading step.
- **sanitize**: Cleans and sanitizes text data within the DataFrame, preparing it for further processing or analysis.
- **fetch_remote_data**: Retrieves data from remote sources, supporting various protocols and formats for seamless integration into the data pipeline.
- **array2hdf5**: Converts NumPy arrays to HDF5 format, offering an efficient storage solution for large numerical datasets.
- **save_or_load**: Provides a flexible interface for saving to or loading data from various formats, enhancing data management practices.
- **request_data**: Facilitates data retrieval from web APIs, handling HTTP requests and responses within the data collection process.
- **fancier_downloader**: Enhances data downloading capabilities with progress tracking and error handling, improving the user experience in data acquisition tasks.
- **speed_rowwise_process**: Optimizes row-wise operations on DataFrames, improving performance for custom data processing functions.
- **store_or_retrieve_data**: Simplifies the storage and retrieval of datasets, supporting various storage backends and formats for versatile data handling.
- **enrich_data_spectrum**: Augments the DataFrame with additional data sources or derived features, broadening the analytical potential of the dataset.
- **format_long_column_names**: Automatically formats long column names for better readability and consistency within the DataFrame.
- **summarize_text_columns**: Generates concise summaries for text columns, facilitating a quick overview of textual data.
- **simple_extractive_summary**: Implements a straightforward extractive summarization technique, condensing text data into essential points.
- **handle_datasets_with_hdfstore**: Manages datasets within HDF5 stores, leveraging the HDFStore interface for efficient data organization and access.
- **verify_data_integrity**: Assesses the overall integrity of the dataset, identifying and reporting on missing values, duplicates, and potential outliers.
- **handle_categorical_features**: Detects and converts numerical columns with limited unique values into categorical ones, improving their representation for analysis.
- **convert_date_features**: Identifies and converts date columns to datetime objects, extracting temporal features for enriched data analysis.
- **scale_data**: Standardizes or normalizes numeric columns, preparing them for use in machine learning models by adjusting their scale.
- **inspect_data**: Provides a comprehensive inspection of the DataFrame, highlighting key aspects of data quality and suggesting improvements.
- **handle_outliers_in_data**: Identifies and addresses outliers in numeric columns, applying specified strategies to mitigate their impact on the dataset.
- **handle_missing_data**: Implements various approaches to manage missing data, enhancing the completeness and usability of the dataset.
- **augment_data**: Increases the diversity of the dataset through data augmentation techniques, aiming to improve model robustness and performance.
- **assess_outlier_impact**: Evaluates the influence of outliers on model performance, offering insights into the necessity of outlier treatment.
- **transform_dates**: Automatically detects and transforms date columns into datetime objects, streamlining the preparation of temporal data.
- **apply_bow_vectorization**: Converts text data into a matrix of token counts using the Bag of Words model, facilitating text analysis and modeling.
- **apply_tfidf_vectorization**: Transforms text columns into a matrix of TF-IDF features, highlighting the importance of words within documents.
- **apply_word_embeddings**: Leverages pre-trained word embeddings to convert text data into dense vector representations, capturing semantic relationships.
- **boxcox_transformation**: Applies the Box-Cox transformation to stabilize variance across numeric columns, aiding in data normalization.
- **check_missing_data**: Analyzes and visualizes the distribution of missing data within the DataFrame, offering a clear view of data integrity issues.

Usage
-----
To utilize these utilities, import the desired function from the ``gofast.tools.baseutils`` module:

.. code-block:: python

   from gofast.tools.baseutils import audit_data
   from gofast.tools.baseutils import apply_tfidf_vectorization
   # ... 

These tools are designed to integrate seamlessly into your data processing pipelines, 
offering a familiar and easy-to-use interface for performing complex data transformations 
and analysis.

Applications
------------
The ``gofast.tools.baseutils`` module is suitable for data scientists, analysts, and machine learning 
practitioners seeking efficient and robust methods for data preparation and feature extraction.
 Whether you're working on natural language processing tasks, predictive modeling, or 
 exploratory data analysis, this module provides essential tools to enhance your workflows.

Examples
--------

audit_data
^^^^^^^^^^
Performs a comprehensive audit on the DataFrame, applying various data cleaning 
and preprocessing steps to enhance the data quality for analytical or modeling purposes.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import audit_data

   data = pd.DataFrame({'A': [1, 2, None], 'B': ['x', None, 'z']})
   audited_data, audit_report = audit_data(data)
   print(audit_report)

read_data
^^^^^^^^^
Reads data from a specified file format into a pandas DataFrame, supporting a 
wide range of file types for seamless data ingestion.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import read_data

   filepath = 'path/to/your/data.csv'
   data = read_data(filepath)
   print(data.head())

sanitize
^^^^^^^^
Cleans and sanitizes text data within a DataFrame, removing or replacing 
unwanted characters, spaces, or patterns to prepare text for further analysis.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import sanitize

   data = pd.DataFrame({'text_column': ['This is a sample!  ', ' Another text, with punctuation. ']})
   sanitized_data = sanitize(data, 'text_column')
   print(sanitized_data)

fetch_remote_data
^^^^^^^^^^^^^^^^^
Downloads data from a specified remote URL, offering options to save the 
downloaded data to a local file for offline access and further processing.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import fetch_remote_data

   remote_url = "https://example.com/data.csv"
   fetch_remote_data(remote_url, save_path="downloaded_data.csv")

array2hdf5
^^^^^^^^^^
Converts numpy arrays to HDF5 format for efficient storage and retrieval, 
facilitating the handling of large numerical datasets.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from gofast.tools.baseutils import array2hdf5

   array_data = np.random.rand(100, 10)
   array2hdf5(array_data, filepath='data.h5', dataset_name='my_dataset')

save_or_load
^^^^^^^^^^^^
Provides a flexible mechanism to either save a pandas DataFrame to a file or 
load it from a file, supporting various formats for efficient data management.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import save_or_load

   data = pd.DataFrame({'A': range(5)})
   save_or_load(data, filepath='data.csv', operation='save')
   loaded_data = save_or_load(filepath='data.csv', operation='load')
   print(loaded_data)

request_data
^^^^^^^^^^^^
Performs HTTP requests to retrieve data from web APIs or other online resources,
 supporting GET and POST methods with customizable parameters and headers.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import request_data

   url = 'https://api.example.com/data'
   response_data = request_data(url, method='get', as_json=True)
   print(response_data)

fancier_downloader
^^^^^^^^^^^^^^^^^^
Downloads files from the internet with progress tracking, providing a 
user-friendly interface for monitoring download progress.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import fancier_downloader

   url = "https://example.com/largefile.zip"
   local_filename = "downloaded_file.zip"
   fancier_downloader(url, local_filename)
   print(f"Downloaded {local_filename} successfully.")

speed_rowwise_process
^^^^^^^^^^^^^^^^^^^^^
Applies a function to each row of a DataFrame in parallel, speeding up row-wise 
operations significantly.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import speed_rowwise_process

   data = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
   def my_function(row):
       return row['A'] + row['B']
   processed_data = speed_rowwise_process(data, my_function)
   print(processed_data.head())

store_or_retrieve_data
^^^^^^^^^^^^^^^^^^^^^^
Facilitates storing and retrieving pandas DataFrames in HDF5 format, providing 
an efficient mechanism for handling large datasets.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import store_or_retrieve_data

   data = pd.DataFrame({'A': range(5), 'B': range(5, 10)})
   filepath = 'my_data.h5'
   store_or_retrieve_data(filepath, datasets={'my_dataset': data}, operation='store')
   retrieved_data = store_or_retrieve_data(filepath, operation='retrieve')
   print(retrieved_data['my_dataset'])

enrich_data_spectrum
^^^^^^^^^^^^^^^^^^^^
Enhances the feature space of a DataFrame by generating new features through 
various transformations, aimed at improving model performance.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import enrich_data_spectrum

   data = pd.DataFrame({'A': range(1, 6), 'B': range(10, 15)})
   enriched_data = enrich_data_spectrum(data)
   print(enriched_data.head())

format_long_column_names
^^^^^^^^^^^^^^^^^^^^^^^^
Automatically shortens long column names in a DataFrame to a specified length, 
maintaining readability and compatibility with various data processing tools.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import format_long_column_names

   data = pd.DataFrame(columns=['This_is_a_very_long_column_name_that_needs_shortening', 'ShortName'])
   formatted_data = format_long_column_names(data, max_length=10)
   print(formatted_data.columns)

summarize_text_columns
^^^^^^^^^^^^^^^^^^^^^^
Generates summary statistics for text columns in a DataFrame, including counts
 of unique values, most common values, and their frequencies.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import summarize_text_columns

   data = pd.DataFrame({'TextColumn': ['apple', 'banana', 'apple', 'orange', 'banana', 'banana']})
   summary = summarize_text_columns(data)
   print(summary)

simple_extractive_summary
^^^^^^^^^^^^^^^^^^^^^^^^^
Creates extractive summaries of text data within a DataFrame by selecting key
sentences, useful for quickly understanding the main points in large texts.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import simple_extractive_summary

   data = pd.DataFrame({'TextColumn': ['This is a long text that will be summarized. It contains several sentences that illustrate the main points.']})
   summarized_data = simple_extractive_summary(data, 'TextColumn')
   print(summarized_data)

handle_datasets_with_hdfstore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Saves or retrieves datasets to/from an HDF5 store, providing a versatile and 
efficient way to work with large volumes of structured data.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import handle_datasets_with_hdfstore

   # Create sample data
   df1 = pd.DataFrame({'A': range(5)})
   df2 = pd.DataFrame({'B': range(5, 10)})
   filepath = 'data.h5'

   # Store data
   handle_datasets_with_hdfstore(filepath, {'df1': df1, 'df2': df2}, operation='store')

   # Retrieve data
   datasets = handle_datasets_with_hdfstore(filepath, operation='retrieve')
   print(datasets['df1'])
   print(datasets['df2'])

verify_data_integrity
^^^^^^^^^^^^^^^^^^^^^
Assesses the integrity of the dataset by checking for missing values, duplicates, 
and potential outliers, ensuring the data is clean and ready for analysis.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import verify_data_integrity

   # Sample DataFrame with missing values and duplicates
   data = pd.DataFrame({'A': [1, 2, None, 4, 4], 'B': [None, 2, 3, 4, 4]})
   is_valid, report = verify_data_integrity(data)
   print(f"Data Integrity: {is_valid}")
   print(report)

handle_categorical_features
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Converts numeric columns in a DataFrame to categorical columns based on a 
threshold of unique values, enhancing the representation of data for analysis.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import handle_categorical_features

   data = pd.DataFrame({'A': [1, 2, 1, 3], 'B': [4, 5, 6, 7]})
   updated_data = handle_categorical_features(data, categorical_threshold=3)
   print(updated_data.dtypes)

convert_date_features
^^^^^^^^^^^^^^^^^^^^^
Automatically identifies and converts date columns to datetime format in a 
DataFrame, extracting additional temporal features for enriched analysis.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import convert_date_features

   data = pd.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03']})
   data_with_date_features = convert_date_features(data, date_features=['date'])
   print(data_with_date_features.head())

scale_data
^^^^^^^^^^
Standardizes or normalizes numeric columns in a DataFrame to improve the 
performance of machine learning models.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import scale_data

   data = pd.DataFrame({'A': range(1, 6), 'B': [2.5, 3.5, 5.5, 6.5, 7.5]})
   scaled_data = scale_data(data, method='minmax')
   print(scaled_data)

inspect_data
^^^^^^^^^^^^
Provides a detailed inspection of the DataFrame, identifying data quality 
issues and offering insights for preprocessing.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import inspect_data

   data = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
   inspect_data(data)

handle_outliers_in_data
^^^^^^^^^^^^^^^^^^^^^^^
Detects and manages outliers in numeric columns of a DataFrame, using techniques 
such as clipping or replacement to ensure data quality.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import handle_outliers_in_data

   data = pd.DataFrame({'A': [1, 2, 100, 3, 4], 'B': [20, 30, -100, 40, 50]})
   data_without_outliers = handle_outliers_in_data(data, method='clip')
   print(data_without_outliers)

handle_missing_data
^^^^^^^^^^^^^^^^^^^
Applies various strategies to manage missing values within a DataFrame, such 
as imputation or removal, maintaining the integrity of the dataset.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import handle_missing_data

   data = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
   data_without_missing = handle_missing_data(data, method='fill_mean')
   print(data_without_missing)

augment_data
^^^^^^^^^^^^
Enhances a dataset by generating augmented copies with slight variations, 
increasing the diversity and size of the dataset for model training.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import augment_data

   data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
   augmented_data = augment_data(data, augmentation_factor=2)
   print(augmented_data)

assess_outlier_impact
^^^^^^^^^^^^^^^^^^^^^
Evaluates how outliers affect the predictive performance of models, offering 
insights for better data preprocessing strategies.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import assess_outlier_impact

   # Create sample DataFrame
   data = pd.DataFrame({'Feature1': [1, 2, 3, 100], 'Feature2': [2, 3, 4, -100], 'Target': [0, 1, 0, 1]})
   performance_with_outliers, performance_without_outliers = assess_outlier_impact(data, target_column='Target')
   print(f"Performance with outliers: {performance_with_outliers}")
   print(f"Performance without outliers: {performance_without_outliers}")

transform_dates
^^^^^^^^^^^^^^^
Detects and transforms date columns into datetime format, extracting and 
enhancing temporal features within a DataFrame.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import transform_dates

   # Sample DataFrame with date strings
   data = pd.DataFrame({'date_column': ['2021-01-01', '2021-01-02', '2021-01-03']})
   transformed_data = transform_dates(data, transform=True, return_dt_columns=True)
   print(f"Transformed Date Columns: {transformed_data}")

apply_bow_vectorization
^^^^^^^^^^^^^^^^^^^^^^^
Converts text data into a Bag of Words model, creating a sparse matrix of 
word counts, useful for text analysis and natural language processing tasks.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import apply_bow_vectorization

   # Create a DataFrame with text
   data = pd.DataFrame({'text_column': ['this is a sample', 'another sample text', 'text data processing']})
   bow_data = apply_bow_vectorization(data, text_columns='text_column', max_features=10)
   print(bow_data.head())

apply_tfidf_vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^
Transforms text columns into a matrix of TF-IDF features, highlighting the 
importance of words within documents across a corpus.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import apply_tfidf_vectorization

   # Create a DataFrame with text
   data = pd.DataFrame({'text_column': ['this is a sample', 'another sample text', 'text data processing']})
   tfidf_data = apply_tfidf_vectorization(data, text_columns='text_column', max_features=10)
   print(tfidf_data.head())

apply_word_embeddings
^^^^^^^^^^^^^^^^^^^^^
Utilizes pre-trained word embeddings to convert text into high-dimensional 
vectors, capturing semantic relationships, followed by dimensionality reduction.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import apply_word_embeddings

   # Sample DataFrame with text data
   data = pd.DataFrame({'text_column': ['text analysis', 'word embeddings', 'dimensionality reduction']})
   embedding_data = apply_word_embeddings(data, text_columns='text_column', embedding_file_path='path/to/embeddings.bin', n_components=50)
   print(embedding_data.head())

boxcox_transformation
^^^^^^^^^^^^^^^^^^^^^
Applies the Box-Cox transformation to numeric columns in a DataFrame, 
stabilizing variance and making the data more normally distributed.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import boxcox_transformation

   # Create a DataFrame with skewed data
   data = pd.DataFrame({'skewed_feature': [0.1, 0.5, 0.9, 2, 3, 4, 5]})
   transformed_data, lambda_values = boxcox_transformation(data, columns=['skewed_feature'], adjust_non_positive='adjust')
   print(transformed_data)
   print(lambda_values)

check_missing_data
^^^^^^^^^^^^^^^^^^
Analyzes a DataFrame for missing values, providing a summary of missing data by 
column and optionally visualizing the distribution.

.. code-block:: python

   import pandas as pd
   from gofast.tools.baseutils import check_missing_data

   # Sample DataFrame with missing values
   data = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None], 'C': [None, None, None]})
   missing_stats = check_missing_data(data, view=True)
   print(missing_stats)
