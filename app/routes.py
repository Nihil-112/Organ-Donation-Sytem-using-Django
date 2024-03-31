from flask import Flask, render_template, request
from model import PricePredictionModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        area = int(request.form['area'])
        stories = int(request.form['stories'])
        mainroad = int(request.form['mainroad'])
        guestroom = int(request.form['guestroom'])
        basement = int(request.form['basement'])
        hotwaterheating = int(request.form['hotwaterheating'])
        airconditioning = int(request.form['airconditioning'])
        parking = int(request.form['parking'])
        prefarea = int(request.form['prefarea'])
        furnishingstatus = int(request.form['furnishingstatus'])

        input_data = [bedrooms, bathrooms, area, stories, mainroad, guestroom, basement, hotwaterheating,
                      airconditioning, parking, prefarea, furnishingstatus]

        # Load and preprocess data
        data_path = "Housing.csv"  # Provide the path to your dataset
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

        # Train the model using the preprocessed DataFrame
        X = df[['area','bedrooms','bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
                'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
        y = df['price']
        model = LinearRegression()
        model.fit(X, y)

        # Predict the house price
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data_reshaped)
        predicted_price = model.predict(input_data_scaled)[0]

        return render_template('prediction.html', prediction=predicted_price)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
