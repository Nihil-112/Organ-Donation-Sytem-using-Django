#prediction_model_2.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression


class PricePredictionModel:
    @staticmethod
    def preprocess_data(data_path):
        df = pd.read_csv(data_path)

        # Encode categorical columns
        categorical_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea",
                               "furnishingstatus"]
        df[categorical_columns] = df[categorical_columns].apply(LabelEncoder().fit_transform)

        # Scale numerical columns
        numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
                             'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
        scaler = MinMaxScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        return df, scaler

    @staticmethod
    def train_model(df):
        X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
                'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
        y = df['price']
        model = LinearRegression()
        model.fit(X, y)
        return model

    @staticmethod
    def predict_price(model, input_data, scaler):
        # Reshape input_data to ensure it's a 2D array
        input_data_reshaped = np.array(input_data).reshape(1, -1)

        # Pad input_data with zeros if it has fewer than 12 features
        input_data_padded = np.hstack([input_data_reshaped, np.zeros((1, 12 - input_data_reshaped.shape[1]))]) if \
        input_data_reshaped.shape[1] < 12 else input_data_reshaped

        input_data_scaled = scaler.transform(input_data_padded)
        prediction = model.predict(input_data_scaled)
        return prediction[0]


# Example usage:
if __name__ == "__main__":
    # Load and preprocess data
    data_path = "app/Housing.csv"  # Provide the path to your dataset
    df, scaler = PricePredictionModel.preprocess_data(data_path)  # Unpack the tuple

    # Train the model using the preprocessed DataFrame
    model = PricePredictionModel.train_model(df)

    # Example prediction
    input_data = [3, 2, 1800, 1, 0, 1, 0, 1, 1, 1]  # Example input data with fewer than 12 features
    predicted_price = PricePredictionModel.predict_price(model, input_data, scaler)
    print("Predicted Price:", predicted_price)
