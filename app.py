from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler


load_dotenv()

app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv('daily_weather_data.csv')

# Convert date to datetime object
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Features we'll use for prediction (excluding tmin and tmax)
features = ['Latitude', 'Longitude', 'wdir', 'wspd', 'pres']

# Drop rows with any NaN values in the features or target
data.dropna(subset=features + ['tavg'], inplace=True)


# Prepare the data for training
X = data[features]
y = data['tavg']


# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
coefficients = model.coef_

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form['city']

        # Get city details from the dataset
        city_data = data[data['city'].str.lower() == city.lower()]
        if city_data.empty:
            error = f"Sorry, we don't have data for {city}."
            return render_template('index.html', error=error)

        # Use the latest record for the city
        latest_entry = city_data.sort_values(by='date').iloc[-1]

        # Prepare input features
        # Prepare input features as a DataFrame with feature names
        input_data = {
            'Latitude': [latest_entry['Latitude']],
            'Longitude': [latest_entry['Longitude']],
            'wdir': [latest_entry['wdir']],
            'wspd': [latest_entry['wspd']],
            'pres': [latest_entry['pres']]
        }

        input_features = pd.DataFrame(input_data)


        # Predict the temperature
        predicted_temp = model.predict(input_features)[0]

        # Fetch actual temperature from OpenWeather API
        api_key = os.getenv('API_KEY')
        if not api_key:
            error = "API key not found. Please set your OpenWeather API key."
            return render_template('index.html', error=error)

        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(weather_url)
        if response.status_code == 200:
            actual_temp = response.json()['main']['temp']
            temp_difference = abs(predicted_temp - actual_temp)
        else:
            actual_temp = None
            temp_difference = None

        zipped_features = zip(features, coefficients.round(4))

        return render_template('index.html',
                               city=city.title(),
                               predicted_temp=round(predicted_temp, 2),
                               actual_temp=actual_temp if actual_temp is not None else 'N/A',
                               temp_difference=round(temp_difference, 2) if temp_difference is not None else 'N/A',
                               zipped_features=zipped_features,
                               coefficients=coefficients.round(4),
                               algorithm='Linear Regression')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
