# Dengue Prediction Models

![Header Image](dengue.png)


https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/

The project involves analyzing Dengue fever dataset which is segmented by location: San Juan and Iquitos.

## Badges

![GitHub license](https://img.shields.io/github/license/JMartinArocha/DenguePredictionModels.svg)
![Python version](https://img.shields.io/badge/python-3.x-blue.svg)
![last-commit](https://img.shields.io/github/last-commit/JMartinArocha/DenguePredictionModels)
![issues](https://img.shields.io/github/issues/JMartinArocha/DenguePredictionModels)
![commit-activity](https://img.shields.io/github/commit-activity/m/JMartinArocha/DenguePredictionModels)
![repo-size](https://img.shields.io/github/repo-size/JMartinArocha/DenguePredictionModels)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the scripts, it's essential to install the necessary Python packages. The project has a `requirements.txt` file listing all the dependencies. You can install these using `pip`. 

Note: The commands below are intended to be run in a Jupyter notebook environment, where the `!` prefix executes shell commands. If you're setting up the project in a different environment, you may omit the `!` and run the commands directly in your terminal.

```bash
!pip3 install --upgrade pip
!pip3 install -r requirements.txt
```

## Importing Shared Utilities

The project utilizes a shared Python utility script hosted on GitHub Gist. This script, `ml_utilities.py`, contains common functions and helpers used across various parts of the project. To ensure you have the latest version of this utility script, the project includes a step to download it directly from GitHub Gist before importing and using its functions.

Below is the procedure to fetch and save the `ml_utilities.py` script programmatically:

## Data Loading and Preprocessing

These datasets are hosted on GitHub, ensuring easy access and version control. Here's how we programmatically load these datasets directly into our project using pandas, a powerful data manipulation library in Python. Additionally, we use custom utility functions to quickly inspect the loaded data.

### Fetching Data

Datasets are stored as CSV files on GitHub. We have separate URLs for training features, training labels, and test features. This setup facilitates direct loading of the datasets into pandas DataFrames for analysis.

### Data Segregation

Given the geographical differences between San Juan and Iquitos, we segment the data accordingly to tailor our analysis and predictive models to each location's unique characteristics.

### Initial Data Inspection

Utilizing custom utility functions from `ml_utilities.py`, we perform an initial inspection of the datasets to understand their structure, missing values, and potential preprocessing needs.

## Data Preparation

Preparing the data correctly is a critical step in any data analysis workflow. For the Dengue fever dataset, this involves several key processes:

### Handling Missing Values

Missing data can significantly impact the performance of predictive models. To address this, we use forward filling (`ffill`) to propagate the last valid observation forward.

### Merging Labels with Features

For both San Juan and Iquitos datasets, we merge the total dengue fever case numbers (our labels) with the feature sets. This integration allows us to build models that predict the total cases based on various environmental and temporal features.

### Data Normalization

Normalization is a technique used to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. This process is essential, especially for datasets with features that vary in magnitudes, units, and range.

## Graphical Methods for Feature Selection

An essential part of the data analysis process involves selecting the right features that contribute most significantly to the target variable. For our dengue fever prediction model, we employ graphical methods to visualize the correlations between different features.

### Correlation Heatmaps

Correlation heatmaps are powerful tools for visualizing the strength and direction of the relationships between the features. By plotting these for both San Juan and Iquitos datasets, we can easily identify highly correlated variables, which might influence our feature selection process.

These visualizations assist in discerning which features to include in our predictive models, aiming to enhance model accuracy and interpretability.

## Visual Analysis of Feature Correlations

To further refine our feature selection process, we perform a visual analysis of how different features correlate with the total dengue fever cases in both San Juan and Iquitos. This step is crucial for identifying the features that may have a significant predictive power for our models.

### Feature Correlation with Total Dengue Cases

For both locales, we generate horizontal bar charts that rank features based on their correlation with the total cases. This visualization allows us to quickly identify which variables are most strongly associated with dengue fever outbreaks, providing a clear direction for feature prioritization.

- **San Juan**: The chart highlights the features with the strongest positive or negative correlation with dengue cases, suggesting their potential influence on the disease's transmission dynamics in the area.

- **Iquitos**: Similarly, the chart for Iquitos showcases the features that may play a critical role in predicting dengue cases, reflecting the unique environmental and temporal patterns affecting the region.

These insights are invaluable for building focused and effective predictive models that consider the most relevant factors influencing dengue fever spread.

## Advanced Feature Relationship Analysis with Pair Plots

Understanding the multidimensional relationships between features can provide deeper insights into the data and help identify complex interactions that might influence dengue fever cases. To this end, we employ pair plots that allow us to visualize the distribution and relationships of feature pairs across different levels of dengue fever occurrences.

### Pair Plots for Feature Comparison

Pair plots are especially useful for spotting trends, correlations, and potential clusters within the data. By generating pair plots for unique combinations of two features and coloring the data points based on dengue case counts, we can visually assess:

- How feature pairs correlate with each other.
- The distribution of data points across different levels of dengue cases.
- Potential outliers or clusters that may warrant further investigation.

This approach is applied separately to the datasets of San Juan and Iquitos to account for geographical and environmental differences that may affect the features' relationships with dengue fever cases.

## Feature Selection with SelectKBest

Feature selection is a crucial step in the data preprocessing phase, especially in datasets with a large number of features. It helps in reducing overfitting, improving accuracy, and increasing model interpretability. For this project, we employ `SelectKBest` from the scikit-learn library to select the most relevant features that contribute to predicting dengue fever cases.

### Applying SelectKBest

`SelectKBest` works by selecting the top `k` features that have the strongest relationship with the target variable, according to a given statistical test. In our case, we use the ANOVA F-value (`f_classif`) for feature scoring.

### Process

1. **Separate the Dataset**: We first exclude non-numeric and target columns from our features (`X`) and define our target variable (`y`).
2. **Feature Selection**: We then apply `SelectKBest` to determine the top `k` features most correlated with the total dengue cases.
3. **Review Selected Features**: Finally, we review the selected features to understand which variables `SelectKBest` identified as most predictive.

This approach provides a data-driven basis for feature selection, streamlining the modeling process.

## Data Splitting: Train, Test, and Validation Sets

To build a robust predictive model, it's crucial to divide our dataset into separate subsets: training, validation, and testing. This strategy allows us to train the model, tune its hyperparameters, and evaluate its performance on unseen data, thus avoiding overfitting and ensuring that our model generalizes well to new data.

### Selecting Features

Based on previous analysis and feature selection techniques, we've identified a subset of features that are highly relevant for predicting the total cases of dengue fever. These features include specific humidity, dew point temperature, average temperature, and minimum temperature.

### Splitting the Dataset

1. **Training Set**: Used to train the model.
2. **Validation Set**: Used to fine-tune model hyperparameters and prevent overfitting.
3. **Testing Set**: Used to evaluate the final model's performance on unseen data.

This methodical splitting ensures that we can assess the model's predictive power accurately.

## Model Training and Hyperparameter Tuning

In this phase of the project, we focus on training a RandomForestRegressor model, a versatile and powerful ensemble method known for its high accuracy and robustness. Given the complexity of our dataset, fine-tuning the model's hyperparameters is crucial to achieve optimal performance.

### Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV

We utilize two strategies for hyperparameter tuning:

- **GridSearchCV**: Exhaustively searches over a specified parameter grid, evaluating model performance for each combination of parameters to identify the best set.
- **RandomizedSearchCV**: Randomly selects a specified number of parameter combinations to test, offering a more computationally efficient alternative to GridSearchCV with the potential for equally effective results.

### Visualizing Feature Importance

Understanding which features have the most significant impact on predictions can provide valuable insights. We visualize the feature importance as determined by the RandomForestRegressor, helping to highlight the variables most influential to our model's predictions.

## Training a DecisionTreeRegressor

After exploring ensemble methods like RandomForestRegressor, we now turn our focus to a more interpretable model, the DecisionTreeRegressor. Decision trees provide clear insight into how decisions are made by the model, making them invaluable for understanding feature influence on predictions.

### Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV

For the DecisionTreeRegressor, hyperparameter tuning is crucial for preventing overfitting and ensuring optimal model complexity.

- **GridSearchCV** is used to exhaustively search through a predefined hyperparameter space, ensuring the best combination is selected based on cross-validated performance.
  
- **RandomizedSearchCV** offers a computationally efficient alternative, randomly sampling from a distribution of hyperparameters to find an effective combination within fewer iterations.

### Feature Importance Visualization

One of the key benefits of decision tree models is their ability to quantify the importance of each feature in making predictions. This can be visualized to understand which features have the most significant impact on the model's decisions.

## Training a GradientBoostingRegressor

The GradientBoostingRegressor is an advanced ensemble technique that combines multiple weak predictive models, typically decision trees, to create a robust model with improved accuracy and performance. It's particularly useful for regression tasks due to its ability to minimize errors through gradient boosting.

### Hyperparameter Tuning with GridSearchCV

To optimize our GradientBoostingRegressor model, we utilize GridSearchCV for hyperparameter tuning. This method systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance based on our data.

### Key Hyperparameters:

- `n_estimators`: Number of boosting stages to be run.
- `max_features`: The number of features to consider when looking for the best split.
- `max_depth`: Maximum depth of the individual regression estimators.
- `learning_rate`: Rate at which the contribution of each tree shrinks.

### Feature Importance Visualization

After training, we analyze the feature importance derived from the model to identify which features most significantly impact the target variable. This insight is crucial for understanding the driving factors behind our predictions.

## Hyperparameter Tuning with RandomizedSearchCV for GradientBoostingRegressor

To further refine our model, we employ RandomizedSearchCV, an efficient method for hyperparameter tuning that randomly samples from defined distributions. This approach is particularly useful when dealing with a large hyperparameter space or when aiming to reduce computational demand without significantly compromising the quality of the model.

### Defining Hyperparameter Distributions

Unlike GridSearchCV, which exhaustively searches through all the possible combinations, RandomizedSearchCV samples from distributions of hyperparameters, offering both flexibility and efficiency. This method is especially effective for tuning models like GradientBoostingRegressor where certain hyperparameters, such as `n_estimators` and `learning_rate`, can have a substantial impact on model performance.

### Model Training and Feature Importance Visualization

After identifying the optimal set of hyperparameters, we train the GradientBoostingRegressor model and visualize its feature importance. This step not only confirms the model's predictive capabilities but also provides insights into which features are most influential in determining the target variable.

## Model Performance Comparison

In our analysis, we've trained multiple regression models, utilizing both GridSearchCV and RandomizedSearchCV for hyperparameter tuning. To understand the precision of our results and compare model performances, we employ graphical representations of key metrics.

### Approach

- We compile a list of models including RandomForestRegressor and GradientBoostingRegressor variants, tuned via GridSearch and RandomizedSearch, as well as DecisionTreeRegressor models.
- For each model, we generate a regression evaluation report that includes metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R^2) values.
- These metrics are collected and compared across all models, providing a comprehensive view of their performance.

### Visualization

- Utilizing a custom function, we plot the comparison of these metrics for all models, adjusting the scale for better visual interpretation.
- This comparative analysis helps in identifying the models that perform best on our dataset based on the selected metrics.

# Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Please refer to the CONTRIBUTING.md for more information.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Authors
Jose Martin de la Fuente Arocha - JMartinArocha






