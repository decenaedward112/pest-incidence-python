from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask_cors import CORS  # Import the CORS module

app = Flask(__name__)

# Enable CORS for a specific origin (localhost:3000 in this case)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load the dataset
data = pd.read_csv('SIYASATv2.csv')

# Define features and targets
features = [
    'crop_age_in_days', 
    'crop_generation', 
    'field_area_in_hectares',
    'temperature_celsius', 
    'relative_humidity_percent',  # Using the correct name from the dataset
    'rainfall_mm'
]
target_pest_population = 'pest_population'
target_yield_loss = 'yield_loss_percent'

# Split data into training and test sets
X_train, X_test, y_train_pest, y_test_pest = train_test_split(
    data[features], 
    data[target_pest_population], 
    test_size=0.2, 
    random_state=42
)

y_train_yield, y_test_yield = train_test_split(
    data[target_yield_loss], 
    test_size=0.2, 
    random_state=42
)

# Train Random Forest model for pest population prediction
pest_model = RandomForestRegressor(n_estimators=100, random_state=42)
pest_model.fit(X_train, y_train_pest)

# Create quadratic features for the yield loss prediction
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a linear regression model on the quadratic features
yield_model = LinearRegression()
yield_model.fit(X_train_poly, y_train_yield)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the user
    input_data = request.get_json()

    crop_age_in_days = input_data.get('crop_age_in_days', 0)
    crop_generation = input_data.get('crop_generation', 0)
    field_area_in_hectares = input_data.get('field_area_in_hectares', 0)
    temperature_celsius = input_data.get('temperature_celsius', 0)
    relative_humidity_percent = input_data.get('relative_humidity_percent', 0)  # Corrected feature name
    rainfall_mm = input_data.get('rainfall_mm', 0)

    # Create a DataFrame for the input
    new_data = pd.DataFrame({
        'crop_age_in_days': np.arange(crop_age_in_days, crop_age_in_days + 150),
        'crop_generation': [crop_generation] * 150,
        'field_area_in_hectares': [field_area_in_hectares] * 150,
        'temperature_celsius': [temperature_celsius] * 150,
        'relative_humidity_percent': [relative_humidity_percent] * 150,  # Corrected feature name
        'rainfall_mm': [rainfall_mm] * 150
    })

    # Transform the new data using quadratic features for yield loss prediction
    new_data_poly = poly.transform(new_data)

    # Predict pest incidence and yield loss for the next 150 days
    pest_predictions_per_day = pest_model.predict(new_data)
    yield_predictions_per_day = yield_model.predict(new_data_poly)

    # Generate per-hour pest predictions
    pest_predictions_per_hour = np.repeat(pest_predictions_per_day / 24, 24)

    # Evaluate the model on the test set for MAE
    y_pred_yield = yield_model.predict(X_test_poly)
    mae_yield = mean_absolute_error(y_test_yield, y_pred_yield)

    # Actual vs predicted yield loss percent
    actual_vs_predicted_yield_loss = {
        "x_axis": list(range(len(y_test_yield))),  # X-axis: Sample indices
        "actual_yield_loss_percent": y_test_yield.tolist(),  # Y-axis: Actual yield loss
        "predicted_yield_loss_percent": y_pred_yield.tolist()  # Y-axis: Predicted yield loss
    }

    # Return all the required information including x-axis data
    return jsonify({
        'pest_incidence_per_hour': {
            "x_axis": list(range(len(pest_predictions_per_hour))),  # X-axis: Hours
            "y_axis": pest_predictions_per_hour.tolist()  # Y-axis: Pest incidence per hour
        },
        'pest_incidence_per_day': {
            "x_axis": list(range(150)),  # X-axis: Days (0 to 150)
            "y_axis": pest_predictions_per_day.tolist()  # Y-axis: Pest incidence per day
        },
        'actual_vs_predicted_yield_loss': actual_vs_predicted_yield_loss,
        'mean_absolute_error_yield': mae_yield
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)