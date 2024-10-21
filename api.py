from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://192.168.1.6:3000"]}})

data = pd.read_csv('SIYASATv2.csv')

features = [
    'crop_age_in_days', 
    'crop_generation', 
    'field_area_in_hectares',
    'temperature_celsius', 
    'relative_humidity_percent', 
    'rainfall_mm'
]
target_pest_population = 'pest_population'
target_yield_loss = 'yield_loss_percent'

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

pest_model = RandomForestRegressor(n_estimators=100, random_state=42)
pest_model.fit(X_train, y_train_pest)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

yield_model = LinearRegression()
yield_model.fit(X_train_poly, y_train_yield)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    crop_age_in_days = input_data.get('crop_age_in_days', 0)
    crop_generation = input_data.get('crop_generation', 0)
    field_area_in_hectares = input_data.get('field_area_in_hectares', 0)
    temperature_celsius = input_data.get('temperature_celsius', 0)
    relative_humidity_percent = input_data.get('relative_humidity_percent', 0)
    rainfall_mm = input_data.get('rainfall_mm', 0)

    new_data = pd.DataFrame({
        'crop_age_in_days': np.arange(crop_age_in_days, crop_age_in_days + 147),
        'crop_generation': [crop_generation] * 147,
        'field_area_in_hectares': [field_area_in_hectares] * 147,
        'temperature_celsius': [temperature_celsius] * 147,
        'relative_humidity_percent': [relative_humidity_percent] * 147,
        'rainfall_mm': [rainfall_mm] * 147
    })
    pest_predictions_per_day = pest_model.predict(new_data)

    weekly_pest_incidence = [pest_predictions_per_day[i * 7:(i + 1) * 7].sum() for i in range(21)]  

    weekly_implications = []  
    first_change_week = None 
    percentage_change = 0
    change_direction = ""

    for i in range(len(weekly_pest_incidence)):
        week_number = i + 1 
        if i == 0:
            implication = f"Week {week_number}: Initial week."
        else:
            change = weekly_pest_incidence[i] - weekly_pest_incidence[i - 1]
            if first_change_week is None and change != 0:  
                first_change_week = week_number
                percentage_change = (change / weekly_pest_incidence[i - 1]) * 100 if weekly_pest_incidence[i - 1] != 0 else 0
                change_direction = "increase" if change > 0 else "decrease"
            
            if change > 0:
                implication = f"Week {week_number}: Pest population increased by {change:.2f} compared to the previous week."
            elif change < 0:
                implication = f"Week {week_number}: Pest population decreased by {-change:.2f} compared to the previous week."
            else:
                implication = f"Week {week_number}: Pest population remained stable compared to the previous week."
        weekly_implications.append(implication)

    y_pred_yield = yield_model.predict(X_test_poly)
    mae_yield = mean_absolute_error(y_test_yield, y_pred_yield)

    actual_vs_predicted_yield_loss = {
        "x_axis": list(range(len(y_test_yield))), 
        "actual_yield_loss_percent": y_test_yield.tolist(), 
        "predicted_yield_loss_percent": y_pred_yield.tolist() 
    }

    response = {
        'pest_incidence_per_week': {
            "xaxis": list(range(1, 22)), 
            "yaxis": weekly_pest_incidence,
            "implications": weekly_implications
        },
        'actual_vs_predicted_yield_loss': actual_vs_predicted_yield_loss,
        'mean_absolute_error_yield': mae_yield,
        'first_change': {
            "week": first_change_week if first_change_week else "No change",  
            "change_direction": change_direction if first_change_week else None,
            "percentage_change": abs(percentage_change) if first_change_week else None  
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


