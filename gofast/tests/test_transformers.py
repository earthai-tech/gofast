# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:18:13 2023

@author: Daniel
"""

import numpy as np
import pandas as pd
# import unittest
import pytest
from gofast.transformers import SequentialBackwardSelection, KMeansFeaturizer

from sklearn.datasets import make_classification
from sklearn.model_selection import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
from gofast.transformers import ( 
    StratifiedWithCategoryAdder, 
    StratifiedUsingBaseCategory, 
    CategorizeFeatures, 
    CombinedAttributesAdder, 
    DataFrameSelector,
    FrameUnion,
    TextFeatureExtractor,
    FeatureSelectorByModel,
    DateFeatureExtractor,
    DimensionalityReducer,
    PolynomialFeatureCombiner,
    CategoricalEncoder2,
    CategoricalEncoder, 
    FeatureScaler,
    MissingValueImputer,
    MissingValueImputer2,
    ColumnSelector,
    ColumnSelector2,
    LogTransformer,
    TimeSeriesFeatureExtractor,
    CategoryFrequencyEncoder,
    DateTimeCyclicalEncoder,
    LagFeatureGenerator,
    DifferencingTransformer,
    MovingAverageTransformer,
    CumulativeSumTransformer,
    SeasonalDecomposeTransformer,
    FourierFeaturesTransformer,
    TrendFeatureExtractor,
    ImageResizer,
    ImageNormalizer,
    ImageToGrayscale,
    ImageAugmenter,
    ImageChannelSelector,
    ImageFeatureExtractor,
    ImageEdgeDetector,
    ImageHistogramEqualizer,
    ImagePCAColorAugmenter,
    ImageBatchLoader, 
)

def test_seasonal_decompose_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.random.randn(100)})
    
    # Initialize SeasonalDecomposeTransformer
    transformer = SeasonalDecomposeTransformer(model='additive', freq=12)
    
    # Fit and transform the DataFrame
    decomposed = transformer.fit_transform(X)
    
    # Check that seasonal decomposition is applied correctly
    assert 'seasonal' in decomposed.columns
    assert 'trend' in decomposed.columns
    assert 'resid' in decomposed.columns

# Add more test cases for SeasonalDecomposeTransformer if needed

def test_fourier_features_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'time': np.arange(100)})
    
    # Initialize FourierFeaturesTransformer
    transformer = FourierFeaturesTransformer(periods=[12, 24])
    
    # Fit and transform the DataFrame
    fourier_features = transformer.fit_transform(X)
    
    # Check that Fourier features are generated correctly
    assert 'sin_12' in fourier_features.columns
    assert 'cos_12' in fourier_features.columns
    assert 'sin_24' in fourier_features.columns
    assert 'cos_24' in fourier_features.columns

# Add more test cases for FourierFeaturesTransformer if needed

def test_trend_feature_extractor():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'time': np.arange(100), 
                      'value': np.random.randn(100)})
    
    # Initialize TrendFeatureExtractor
    transformer = TrendFeatureExtractor(order=1)
    
    # Fit and transform the DataFrame
    trend_features = transformer.fit_transform(X[['time']])
    
    # Check that trend features are extracted correctly
    assert 'trend_1' in trend_features.columns

# Add more test cases for TrendFeatureExtractor if needed

def test_image_resizer():
    # Create a sample image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageResizer
    resizer = ImageResizer(output_size=(128, 128))
    
    # Transform the image
    resized_image = resizer.transform(image)
    
    # Check that the image is resized to the specified size
    assert resized_image.shape == (128, 128, 3)

# Add more test cases for ImageResizer if needed

def test_image_normalizer():
    # Create a sample image with pixel values in [0, 255]
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Initialize ImageNormalizer
    normalizer = ImageNormalizer()
    
    # Transform the image
    normalized_image = normalizer.transform(image)
    
    # Check that pixel values are scaled to [0, 1]
    assert np.all(np.logical_and(normalized_image >= 0, normalized_image <= 1))

# Add more test cases for ImageNormalizer if needed

def test_image_to_grayscale():
    # Create a sample color image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageToGrayscale
    converter = ImageToGrayscale(keep_dims=True)
    
    # Transform the image to grayscale
    grayscale_image = converter.transform(image)
    
    # Check that the image is converted to grayscale
    assert grayscale_image.shape == (256, 256, 1)

# Add more test cases for ImageToGrayscale if needed

def test_image_augmenter():
    # Create a sample image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageAugmenter
    augmenter = ImageAugmenter()
    
    # Transform the image using data augmentation techniques
    augmented_image = augmenter.transform(image)
    
    # Check that the augmented image is of the same shape
    assert augmented_image.shape == (256, 256, 3)



def test_log_transformer():
    # Create a sample DataFrame with numeric features
    X = pd.DataFrame({'income': [50000, 80000, 120000]})
    
    # Initialize LogTransformer
    transformer = LogTransformer(numeric_features=['income'], epsilon=1e-6)
    
    # Fit and transform the DataFrame
    X_transformed = transformer.fit_transform(X)
    
    # Check that log transformation is applied correctly
    assert np.allclose(X_transformed['income'], np.log(X['income'] + 1e-6))

# Add more test cases for LogTransformer if needed

def test_time_series_feature_extractor():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'time_series': np.random.rand(100)})
    
    # Initialize TimeSeriesFeatureExtractor
    extractor = TimeSeriesFeatureExtractor(rolling_window=5)
    
    # Fit and transform the DataFrame
    features = extractor.fit_transform(X)
    
    # Check the shape of the extracted features
    assert features.shape == (100, 5)  # Five rolling statistics computed

# Add more test cases for TimeSeriesFeatureExtractor if needed

def test_category_frequency_encoder():
    # Create a sample DataFrame with categorical data
    X = pd.DataFrame({'brand': ['apple', 'apple', 'samsung', 'samsung', 'nokia']})
    
    # Initialize CategoryFrequencyEncoder
    encoder = CategoryFrequencyEncoder(categorical_features=['brand'])
    
    # Fit and transform the DataFrame
    encoded_features = encoder.fit_transform(X)
    
    # Check that categorical encoding is applied correctly
    assert encoded_features['brand'].equals(pd.Series([0.4, 0.4, 0.4, 0.4, 0.2], name='brand'))

# Add more test cases for CategoryFrequencyEncoder if needed

def test_date_time_cyclical_encoder():
    # Create a sample DataFrame with datetime data
    X = pd.DataFrame({'timestamp': pd.date_range(start='1/1/2018', periods=24, freq='H')})
    
    # Initialize DateTimeCyclicalEncoder
    encoder = DateTimeCyclicalEncoder(datetime_features=['timestamp'])
    
    # Fit and transform the DataFrame
    encoded_features = encoder.fit_transform(X)
    
    # Check that cyclical encoding is applied correctly
    assert 'timestamp_sin_hour' in encoded_features.columns
    assert 'timestamp_cos_hour' in encoded_features.columns

# Add more test cases for DateTimeCyclicalEncoder if needed

def test_lag_feature_generator():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.arange(100)})
    
    # Initialize LagFeatureGenerator
    generator = LagFeatureGenerator(lags=3)
    
    # Fit and transform the DataFrame
    lag_features = generator.fit_transform(X)
    
    # Check that lag features are generated correctly
    assert 'lag_1' in lag_features.columns
    assert 'lag_2' in lag_features.columns
    assert 'lag_3' in lag_features.columns

# Add more test cases for LagFeatureGenerator if needed

def test_differencing_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.cumsum(np.random.randn(100))})
    
    # Initialize DifferencingTransformer
    transformer = DifferencingTransformer(periods=1)
    
    # Fit and transform the DataFrame
    stationary_data = transformer.fit_transform(X)
    
    # Check that differencing is applied correctly
    assert 'value' in stationary_data.columns
    assert stationary_data['value'].iloc[0] == 0

# Add more test cases for DifferencingTransformer if needed

def test_moving_average_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.random.randn(100)})
    
    # Initialize MovingAverageTransformer
    transformer = MovingAverageTransformer(window=5)
    
    # Fit and transform the DataFrame
    moving_avg = transformer.fit_transform(X)
    
    # Check that moving average is computed correctly
    assert 'value' in moving_avg.columns

# Add more test cases for MovingAverageTransformer if needed

def test_cumulative_sum_transformer():
    # Create a sample DataFrame with time series data
    X = pd.DataFrame({'value': np.random.randn(100)})
    
    # Initialize CumulativeSumTransformer
    transformer = CumulativeSumTransformer()
    
    # Fit and transform the DataFrame
    cum_sum = transformer.fit_transform(X)
    
    # Check that cumulative sum is computed correctly
    assert 'value' in cum_sum.columns
    assert cum_sum['value'].iloc[-1] == X['value'].sum()

# Test SequentialBackwardSelection
def test_sequential_backward_selection():
    # Generate a sample dataset for testing
    data = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['target'] = np.random.choice([0, 1], size=100)

    # Create a SequentialBackwardSelection instance
    sbs = SequentialBackwardSelection(n_features_to_select=3)

    # Fit and transform the data
    X_selected = sbs.fit_transform(df.drop('target', axis=1), df['target'])

    # Ensure that the transformed data has the expected shape
    assert X_selected.shape == (df.shape[0], 3)

# Test KMeansFeaturizer
def test_kmeans_featurizer():
    # Generate a sample dataset for testing
    data = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])

    # Create a KMeansFeaturizer instance
    kmeans_featurizer = KMeansFeaturizer(n_clusters=3)

    # Fit and transform the data
    X_kmeans = kmeans_featurizer.fit_transform(df)

    # Ensure that the transformed data has the expected shape
    assert X_kmeans.shape == (df.shape[0], 3)

# Run the tests
# if __name__ == "__main__":
#     pytest.main([__file__])


# Test StratifiedWithCategoryAdder
def test_stratified_with_category_adder():
    # Generate a sample dataset for testing
    data = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=100)

    # Create a StratifiedWithCategoryAdder instance
    stratified_adder = StratifiedWithCategoryAdder(base_num_feature='feature1')

    # Transform the data
    df_transformed, _ = stratified_adder.fit_transform(df)

    # Ensure that transformed data has the expected shape
    assert df_transformed.shape == df.shape

# Test StratifiedUsingBaseCategory
def test_stratified_using_base_category():
    # Generate a sample dataset for testing
    data = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=100)

    # Create a StratifiedUsingBaseCategory instance
    stratified_base = StratifiedUsingBaseCategory(base_column='category')

    # Transform the data
    df_transformed, _ = stratified_base.fit_transform(df)

    # Ensure that transformed data has the expected shape
    assert df_transformed.shape == df.shape

# Test CategorizeFeatures
def test_categorize_features():
    # Generate a sample dataset for testing
    data = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    df['category'] = np.random.choice(['A', 'B', 'C'], size=100)

    # Create a CategorizeFeatures instance
    categorizer = CategorizeFeatures(columns=['category'])

    # Transform the data
    df_transformed = categorizer.fit_transform(df)

    # Ensure that transformed data has the expected shape
    assert df_transformed.shape == df.shape

    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])

    # Ensure that the transformed data matches the LabelEncoder encoding
    assert np.array_equal(df_transformed['category'], df['category_encoded'])


# Define your test functions
def test_combined_attributes_adder():
    # Create a sample DataFrame
    data = {
        'lwi': [1, 2, 3, 4, 5],
        'ohmS': [0.1, 0.2, 0.5, 0.4, 0.8],
    }
    df = pd.DataFrame(data)
    
    # Initialize CombinedAttributesAdder
    cobj = CombinedAttributesAdder(attribute_names=['lwi_per_ohmS'])
    
    # Transform the DataFrame
    Xadded = cobj.fit_transform(df)
    
    # Check if the new attribute is added correctly
    assert 'lwi_per_ohmS' in Xadded.columns
    assert np.allclose(Xadded['lwi_per_ohmS'], [10.0, 10.0, 6.0, 10.0, 6.25])

def test_data_frame_selector():
    # Create a sample DataFrame
    data = {
        'numeric1': [1, 2, 3],
        'numeric2': [4, 5, 6],
        'category1': ['A', 'B', 'A'],
    }
    df = pd.DataFrame(data)
    
    # Initialize DataFrameSelector for numeric attributes
    num_selector = DataFrameSelector(attribute_names=['numeric1', 'numeric2'], select_type='num')
    
    # Transform the DataFrame
    X_num = num_selector.fit_transform(df)
    
    # Check if the selected numeric columns are correct
    assert X_num.shape == (3, 2)
    assert np.allclose(X_num, np.array([[1, 4], [2, 5], [3, 6]]))
    
    # Initialize DataFrameSelector for categorical attributes
    cat_selector = DataFrameSelector(attribute_names=['category1'], select_type='cat')
    
    # Transform the DataFrame
    X_cat = cat_selector.fit_transform(df)
    
    # Check if the selected categorical column is correct
    assert X_cat.shape == (3, 1)
    assert np.array_equal(X_cat, np.array([['A'], ['B'], ['A']]))

def test_frame_union():
    # Create a sample DataFrame
    data = {
        'numeric1': [1, 2, 3],
        'numeric2': [4, 5, 6],
        'category1': ['A', 'B', 'A'],
    }
    df = pd.DataFrame(data)
    
    # Initialize FrameUnion with default settings
    frame_union = FrameUnion()
    
    # Transform the DataFrame
    X = frame_union.fit_transform(df)
    
    # Check the shape of the transformed DataFrame
    assert X.shape == (3, 5)  # 2 numeric columns + 3 one-hot encoded categorical columns

    # Initialize FrameUnion with custom settings (e.g., scaling)
    frame_union = FrameUnion(scale=True, encode=False)
    
    # Transform the DataFrame
    X = frame_union.fit_transform(df)
    
    # Check the shape of the transformed DataFrame
    assert X.shape == (3, 2)  # 2 scaled numeric columns
    


def test_text_feature_extractor():
    # Create a sample list of text data
    text_data = ['sample text data', 'another sample text']
    
    # Initialize TextFeatureExtractor
    extractor = TextFeatureExtractor(max_features=500)
    
    # Fit and transform the text data
    features = extractor.fit_transform(text_data)
    
    # Check the shape of the transformed features
    assert features.shape == (2, 500)  # Assuming 500 max features

# Add more test cases for TextFeatureExtractor if needed


def test_date_feature_extractor():
    # Create a sample DataFrame with date columns
    date_data = pd.DataFrame({'date': ['2021-01-01', '2021-02-01']})
    
    # Initialize DateFeatureExtractor
    extractor = DateFeatureExtractor()
    
    # Fit and transform the date data
    features = extractor.fit_transform(date_data)
    
    # Check the shape of the transformed features
    assert features.shape == (2, 3)  # Year, month, and day features added

# Add more test cases for DateFeatureExtractor if needed

def test_feature_selector_by_model():
    # Create a sample dataset
    X = np.random.randn(10, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    
    # Initialize FeatureSelectorByModel with a model (Logistic Regression)
    model = LogisticRegression()
    selector = FeatureSelectorByModel(model)
    
    # Fit and transform the dataset
    X_selected = selector.fit_transform(X, y)
    
    # Check the shape of the selected features
    assert X_selected.shape == (10, 5)  # No feature selection applied

# Add more test cases for FeatureSelectorByModel if needed

def test_categorical_encoder2():
    # Create a sample dataset
    X = [['Category A'], ['Category B'], ['Category A']]
    
    # Initialize CategoricalEncoder2
    enc = CategoricalEncoder2()
    
    # Fit and transform the dataset
    enc.fit(X)
    X_encoded = enc.transform([['Category B'], ['Category C']]).toarray()
    
    # Check the shape of the encoded features
    assert X_encoded.shape == (2, 2)  # Encoded as one-hot

# Add more test cases for CategoricalEncoder2 if needed


def test_categorical_encoder():
    # Create a sample dataset
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    
    # Initialize CategoricalEncoder
    enc = CategoricalEncoder()
    
    # Fit and transform the dataset
    enc.fit(X)
    X_encoded = enc.transform([['Female', 1], ['Male', 4]]).toarray()
    
    # Check the shape of the encoded features
    assert X_encoded.shape == (2, 4)  # Encoded as one-hot

# Add more test cases for CategoricalEncoder if needed


def test_polynomial_feature_combiner():
    # Create a sample dataset
    X = np.arange(6).reshape(3, 2)
    
    # Initialize PolynomialFeatureCombiner
    combiner = PolynomialFeatureCombiner(degree=2)
    
    # Fit and transform the dataset
    X_poly = combiner.fit_transform(X)
    
    # Check the shape of the transformed features
    assert X_poly.shape == (3, 6)  # Polynomial features added

# Add more test cases for PolynomialFeatureCombiner if needed


def test_dimensionality_reducer():
    # Create a sample dataset
    X = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Initialize DimensionalityReducer
    reducer = DimensionalityReducer(n_components=1)
    
    # Fit and transform the dataset
    X_reduced = reducer.fit_transform(X)
    
    # Check the shape of the reduced features
    assert X_reduced.shape == (3, 1)  # Reduced to 1 component

# Add more test cases for DimensionalityReducer if needed


def test_column_selector2():
    # Create a sample DataFrame
    X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    
    # Initialize ColumnSelector2
    selector = ColumnSelector2(column_names=['A', 'B'])
    
    # Fit and transform the DataFrame
    X_selected = selector.fit_transform(X)
    
    # Check that only selected columns are present
    assert list(X_selected.columns) == ['A', 'B']

# Add more test cases for ColumnSelector2 if needed


def test_column_selector():
    # Create a sample DataFrame
    X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    
    # Initialize ColumnSelector
    selector = ColumnSelector(column_names=['A', 'B'])
    
    # Fit and transform the DataFrame
    X_selected = selector.fit_transform(X)
    
    # Check that only selected columns are present
    assert list(X_selected.columns) == ['A', 'B']

# Add more test cases for ColumnSelector if needed


def test_missing_value_imputer2():
    # Create a sample DataFrame with missing values
    X = pd.DataFrame({'age': [25, np.nan, 50], 'income': [50000, 80000, np.nan]})
    
    # Initialize MissingValueImputer2
    imputer = MissingValueImputer2(strategy='mean')
    
    # Fit and transform the DataFrame
    X_imputed = imputer.fit_transform(X)
    
    # Check that missing values are imputed
    assert not X_imputed.isnull().values.any()  # No NaN values should be present

# Add more test cases for MissingValueImputer2 if needed


def test_missing_value_imputer():
    # Create a sample dataset with missing values
    X = np.array([[1, 2], [np.nan, 3], [7, 6]])
    
    # Initialize MissingValueImputer
    imputer = MissingValueImputer(strategy='mean')
    
    # Fit and transform the dataset
    imputer.fit(X)
    X_imputed = imputer.transform(X)
    
    # Check that missing values are imputed
    assert not np.isnan(X_imputed).any()  # No NaN values should be present

# Add more test cases for MissingValueImputer if needed


def test_feature_scaler():
    # Create a sample dataset
    X = np.array([[0, 15], [1, -10]])
    
    # Initialize FeatureScaler
    scaler = FeatureScaler()
    
    # Fit and transform the dataset
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    # Check the shape of the scaled features
    assert X_scaled.shape == (2, 2)  # Scaled to zero mean and unit variance


# Add more test cases for ImageAugmenter if needed

def test_image_channel_selector():
    # Create a sample color image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImageChannelSelector to select the first two channels
    selector = ImageChannelSelector(channels=[0, 1])
    
    # Transform the image to select specific channels
    selected_channels = selector.transform(image)
    
    # Check that the selected channels are of the correct shape
    assert selected_channels.shape == (256, 256, 2)

# Add more test cases for ImageChannelSelector if needed

def test_image_feature_extractor():
    # Create a sample image
    image = np.random.rand(224, 224, 3)
    
    # Initialize ImageFeatureExtractor with a pre-trained model
#xxx TOTP 
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    base_model = VGG16(weights='imagenet', include_top=False)
    # Replace with an actual pre-trained model
    model = Model(inputs=base_model.input, 
                      outputs=base_model.get_layer('block3_pool').output)
    extractor = ImageFeatureExtractor(model=model)
    
    # Transform the image to extract features using the model
    features = extractor.transform(image)
    
    # Check that the extracted features have the expected shape
    assert features.shape == (7, 7, 512)  # Example shape, replace as needed

# Add more test cases for ImageFeatureExtractor if needed

def test_image_edge_detector():
    # Create a sample grayscale image
    image = np.random.rand(256, 256)
    
    # Initialize ImageEdgeDetector with the Sobel method
    detector = ImageEdgeDetector(method='sobel')
    
    # Transform the image to detect edges
    edges = detector.transform(image)
    
    # Check that the edges image is of the same shape
    assert edges.shape == (256, 256)

# Add more test cases for ImageEdgeDetector if needed

def test_image_histogram_equalizer():
    # Create a sample grayscale image
    image = np.random.rand(256, 256)
    
    # Initialize ImageHistogramEqualizer
    equalizer = ImageHistogramEqualizer()
    
    # Transform the image to apply histogram equalization
    equalized_image = equalizer.transform(image)
    
    # Check that the equalized image is of the same shape
    assert equalized_image.shape == (256, 256)

# Add more test cases for ImageHistogramEqualizer if needed

def test_image_pca_color_augmenter():
    # Create a sample color image
    image = np.random.rand(256, 256, 3)
    
    # Initialize ImagePCAColorAugmenter with alpha_std
    augmenter = ImagePCAColorAugmenter(alpha_std=0.1)
    
    # Transform the image to apply PCA color augmentation
    pca_augmented_image = augmenter.transform(image)
    
    # Check that the PCA-augmented image is of the same shape
    assert pca_augmented_image.shape == (256, 256, 3)

# Add more test cases for ImagePCAColorAugmenter if needed

def test_image_batch_loader():
    # Create a list of image paths (replace with actual paths)
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    
    # Initialize ImageBatchLoader with batch size and image loading function
    loader =ImageBatchLoader(batch_size=2, load_image_function=load_image)
    
    # Load images in batches
    for batch_images in loader.transform(image_paths):
        # Check that each batch of images is of the specified batch size
        assert len(batch_images) == 2

# Run the tests
if __name__ == "__main__":
    test_combined_attributes_adder()
    test_data_frame_selector()
    test_frame_union()
    # Add more test cases for CumulativeSumTransformer if needed
    test_log_transformer()
    test_time_series_feature_extractor()
    test_category_frequency_encoder()
    test_date_time_cyclical_encoder()
    test_lag_feature_generator()
    test_differencing_transformer()
    test_moving_average_transformer()
    test_cumulative_sum_transformer()
    # Add more test cases for ImageAugmenter if needed
    test_seasonal_decompose_transformer()
    test_fourier_features_transformer()
    test_trend_feature_extractor()
    test_image_resizer()
    test_image_normalizer()
    test_image_to_grayscale()
    test_image_augmenter()
    test_image_augmenter()
    test_image_channel_selector()
    test_image_feature_extractor()
    test_image_edge_detector()
    test_image_histogram_equalizer()
    test_image_pca_color_augmenter()
    test_image_batch_loader()

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
