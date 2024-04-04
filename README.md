# Hyperspectral Image Classification

This Python project provides a comprehensive pipeline for classifying hyperspectral imagery data. It is designed to preprocess hyperspectral images, apply various classification algorithms, and evaluate the performance of these classifiers. The pipeline utilizes machine learning techniques, including feature selection, normalization, and handling imbalanced datasets, to predict soil types from hyperspectral data.

## Features
 
* **Data Visualization**: Tools for visualizing hyperspectral bands, distribution plots, and spectral signatures to understand data characteristics.
* **Data Preprocessing**: Includes standardization and handling of class imbalance using SMOTE.
* **Modeling**: Supports various machine learning models with a focus on hyperparameter tuning through grid search.
* **Evaluation**: Offers detailed performance evaluation through accuracy metrics, confusion matrices, ROC curves, and classification maps.
* **Scalability**: Pipeline is structured to easily extend for additional models or different datasets.

## Prerequisites

Ensure you have the following libraries installed before running the framework:

* **scipy**
* **numpy**
* **pandas**
* **scikit-learn**
* **matplotlib**
* **xgboost**
* **seaborn**
* **imbalanced-learn**
* **scikit-plot**

## Core Components
1. ### Utility Functions
  * **draw_classification_map()**: Visualizes the classification map of predictions.
  * **distribution_plot():** Plots the distribution of pixel intensities across a single band.
  * **box_plot()**: Displays a box plot for visualizing the distribution of pixel intensities across different classes for a specific band.
  * b**ar_plot()**: Shows the distribution of samples across different classes.
  * **list2array()**: Converts lists to NumPy arrays, facilitating data manipulation.
  * **plot_confusion_matrix()**: Generates a confusion matrix to evaluate model performance.
   * **save_report()**: Saves a comprehensive classification report including metrics, best parameters, and accuracy.
2. ### Data Loading and Visualization
* **Loading Data**: The hyperspectral dataset is loaded from a CSV file.
* **Plotting Functions**: Functions to plot individual bands, spectral signatures, and the ground truth image, aiding in initial data analysis.
3. ### Data Preprocessing
* **Splitting Data**: Stratified splitting of the dataset into training and testing sets.
* **Standardization**: Application of MinMaxScaler for feature scaling.
4. ### Modeling and Evaluation
* **Classifier Definitions**: A collection of machine learning models including KNN, SVM (with RBF and Polynomial kernels), XGBoost, Random Forest, Gradient Boosting, and MLP.
* **Classification Pipeline**: A pipeline that encompasses SMOTE for resampling, grid search for hyperparameter tuning, and the application of classifiers.
*   **Performance Evaluation**: After training, the framework evaluates each model on the test set, generating detailed metrics, confusion matrices, ROC curves, and classification maps.

## Adding New Classifiers
To add a new classifier:

1. Define the classifier and its parameter grid in the classifiers dictionary.
2. Add the classifier to the selected_classifiers list.
3. Execute the pipeline to include the new classifier in the analysis
