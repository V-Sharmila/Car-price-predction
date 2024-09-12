Old Car Price Prediction
This repository contains code for predicting the prices of old cars using machine learning. The project uses a dataset of car prices and features to build a predictive model with a Random Forest Regressor.

Project Overview
The project includes the following steps:

Data Loading: Importing and inspecting the dataset.
Data Preprocessing: Handling missing values, scaling numerical features, and encoding categorical features.
Model Training: Training a Random Forest Regressor on the preprocessed data.
Evaluation: Assessing model performance with Mean Absolute Error (MAE) and Mean Squared Error (MSE).
Files
car_price.csv: The dataset with car features and prices.
car_price_prediction.py: Python script for data preprocessing, model training, and evaluation.
Dependencies
The script requires the following Python packages:

pandas
numpy
scikit-learn
You can install these packages using:
pip install pandas numpy scikit-learn

Usage
Clone the repository:
git clone [your-repository-url]
cd [repository-folder]


python car_price_prediction.py
The script will process the data, train the model, and print the MAE and MSE of the model's predictions.

Code Explanation
Data Loading: Uses pandas to read the dataset and check for missing values.
Data Preprocessing:
Scales numerical features with StandardScaler.
Encodes categorical features using pd.get_dummies().
Model Training: Trains a RandomForestRegressor with 100 estimators.
Evaluation: Calculates MAE and MSE to evaluate model performance.
Results
The script outputs the Mean Absolute Error (MAE) and Mean Squared Error (MSE) for the model's predictions on the test set.
