import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump
import os

try:
    data = pd.read_csv("C:/Users/rutuj/OneDrive/Desktop/RUTUJ/AmesHousing.csv")

    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    data = pd.get_dummies(data)

    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    selected_features = ['Gr Liv Area', 'Overall Qual', 'Garage Cars', 'Year Built']
    
    features = [feature for feature in selected_features if feature in data.columns]
    
    if 'SalePrice' not in data.columns:
        raise ValueError("Target column 'SalePrice' not found in the dataset.")
        
    X = data[features]
    y = data['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    model_path = 'house_price_model.joblib'
    dump(model, model_path)

    print("Model training and saving completed successfully.")
except FileNotFoundError:
    print("The specified file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
