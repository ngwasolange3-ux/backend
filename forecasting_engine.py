import pandas as pd
import numpy as np
import json
import sys
import logging
from statsmodels.tsa.seasonal import STL
import pmdarima as pm
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data (sample data, replace with your CSV)
try:
    data = pd.DataFrame({
        'record_id': ['BB000003', 'BB000003', 'BB000006', 'BB000008', 'BB000009', 'BB000011', 'BB000013', 'BB000014', 'BB000018', 'BB000018', 'BB000021', 'BB000021', 'BB000025', 'BB000027', 'BB000030', 'BB000032', 'BB000034'],
        'donor_id': ['D006011', 'D000744', 'D001516', 'D011573', 'D008804', 'D014664', 'D019464', 'D016142', 'D018869', 'D008548', 'D007227', 'D017439', 'D005889', 'D002303', 'D008699', 'D009870', 'D005774'],
        'donor_age': [60, 38, 37, 59, 20, 58, 37, 24, 19, 38, 42, 35, 47, 46, 7.329401831, 50, 32],
        'donor_gender': ['F', 'M', 'F', 'F', 'F', 'M', 'M', 'F', 'M', 'M', 'UNKNOWN', 'M', 'F', 'M', 'F', 'F', 'F'],
        'blood_type': ['AB+', 'O+', 'B+', 'B+', 'B-', 'AB-', 'A+', 'O-', 'AB+', 'A+', 'B+', 'O-', 'B-', 'O-', 'O+', 'B+', 'O-'],
        'collection_site': ['Douala General', 'Laquintinie', 'District', 'Unknown Site', 'Laquintinie', 'Douala General', 'District', 'Laquintinie', 'District', 'Laquintinie', 'Douala General', 'Laquintinie', 'Unknown Site', 'Douala General', 'Douala General', 'District', 'Laquintinie'],
        'donation_date': ['11/6/2024', '4/3/2025', '12/13/2024', '3/23/2025', '1/1/2025', '9/18/2024', '4/10/2025', '9/10/2024', '8/29/2024', '1/16/2025', '12/4/2024', '4/17/2025', '4/17/2025', '8/17/2024', '11/12/2024', '8/11/2024', '9/17/2024'],
        'expiry_date': ['5/1/2025', '5/22/2025', '10/24/2024', '10/28/2024', '12/20/2024', '10/30/2024', '5/22/2025', '2/23/2025', '6/23/2025', '4/15/2025', '2/27/2025', '11/30/2024', '5/21/2025', '10/13/2024', '4/30/2025', '4/2/2025', '5/31/2025'],
        'collection_volume_ml': [500, 500, 450, 350, 500, 350, 350, 250, 400, 500, 500, 450, 500, 400, 450, 400, 500],
        'hemoglobin_g_dl': [12.54984315, 13.06084991, 13.11664486, 13.37809871, 14.39137232, 13.58146593, 13.36056425, 13.88065163, 14.70614885, 12.78754881, 13.84066369, 13.08898133, 14.54093648, 11.94630273, 14.33432968, 13.58683418, 12.64887417],
        'shelf_life_days': [176, 49, -50, -146, -12, 42, 42, 166, 298, 89, 85, -138, 34, 57, 169, 234, 256],
        'will_expire_early': [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    })
    data.to_csv("blood_stock_data.csv", index=False)
    data = pd.read_csv("blood_stock_data.csv", parse_dates=['donation_date'])
except Exception as e:
    logging.error(f"Error loading data: {e}. Ensure CSV file has 'donation_date' and 'collection_volume_ml' columns.")
    sys.exit(1)

# Filter usable blood and aggregate
try:
    usable_data = data[(data['shelf_life_days'] > 0) & (data['will_expire_early'] == 0)]
    if usable_data.empty:
        raise ValueError("No usable blood data after filtering.")
    series = usable_data.groupby('donation_date')['collection_volume_ml'].sum()
    series.index = pd.to_datetime(series.index)
    date_range = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
    series = series.reindex(date_range, fill_value=0)
    if len(series) < 14:
        raise ValueError("Insufficient data points for forecasting. Need at least 14 days.")
except Exception as e:
    logging.error(f"Data processing error: {e}")
    sys.exit(1)

# 1. STL Decomposition
try:
    stl = STL(series, seasonal=7, robust=True)  # Weekly seasonality
    stl_result = stl.fit()
    seasonal, trend, residual = stl_result.seasonal, stl_result.trend, stl_result.resid
    stl_forecast = trend[-1] + seasonal[-7:]  # Extend trend + repeat last 7 days
    stl_forecast = stl_forecast[:12]  # Limit to 12 days
except Exception as e:
    logging.error(f"STL Error: {e}")
    stl_forecast = np.zeros(12)

# 2. ARIMA
try:
    arima_model = pm.auto_arima(series, seasonal=True, m=7, stepwise=True, suppress_warnings=True)
    arima_forecast = arima_model.predict(n_periods=12)
except Exception as e:
    logging.error(f"ARIMA Error: {e}")
    arima_forecast = np.zeros(12)

# 3. XGBoost
try:
    df = pd.DataFrame({'stock_level': series}, index=series.index)
    df['lag1'] = series.shift(1)
    df['lag2'] = series.shift(2)
    df['day_of_week'] = series.index.dayofweek
    df = df.dropna()
    X = df[['lag1', 'lag2', 'day_of_week']]
    y = df['stock_level']
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_model.fit(X, y)
    last_data = df[['lag1', 'lag2', 'day_of_week']].iloc[-1]
    future_X = []
    last_series = series.values
    for i in range(12):
        day_of_week = (series.index[-1].dayofweek + i + 1) % 7
        lag1 = last_series[-1] if i == 0 else future_X[-1][0]
        lag2 = last_series[-2] if i <= 1 else future_X[-2][0]
        future_X.append([lag1, lag2, day_of_week])
        last_series = np.append(last_series, xgb_model.predict([future_X[-1]])[0])
    future_X = np.array(future_X)
    xgb_forecast = xgb_model.predict(future_X)
except Exception as e:
    logging.error(f"XGBoost Error: {e}")
    xgb_forecast = np.zeros(12)

# 4. TensorFlow LSTM
try:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    seq_length = 7
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    X, y = create_sequences(scaled_data, seq_length)
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    tf_forecast = []
    for _ in range(12):
        pred = model.predict(last_sequence, verbose=0)
        tf_forecast.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = pred[0, 0]
    tf_forecast = scaler.inverse_transform(np.array(tf_forecast).reshape(-1, 1)).flatten()
except Exception as e:
    logging.error(f"TensorFlow Error: {e}")
    tf_forecast = np.zeros(12)

# 5. Ensemble
try:
    ensemble_forecast = (arima_forecast + xgb_forecast + tf_forecast) / 3
except Exception as e:
    logging.error(f"Ensemble Error: {e}")
    ensemble_forecast = np.zeros(12)

# Save results to JSON
try:
    future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=12, freq='D')
    output = {
        'historical': {
            'dates': series.index.strftime('%Y-%m-%d').tolist(),
            'values': series.values.tolist()
        },
        'forecasts': {
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'stl': stl_forecast.tolist(),
            'arima': arima_forecast.tolist(),
            'xgboost': xgb_forecast.tolist(),
            'tensorflow': tf_forecast.tolist(),
            'ensemble': ensemble_forecast.tolist()
        }
    }
    with open('forecast_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    logging.info("Forecasts saved to 'forecast_results.json'")
except Exception as e:
    logging.error(f"Error saving JSON: {e}")
    sys.exit(1)
