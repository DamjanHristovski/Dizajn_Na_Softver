import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from datetime import timedelta
import io
import base64
from matplotlib.figure import Figure
import os
from django.conf import settings

def predict_stock_prices(company_code):
    # Paths to your CSV files
    company_codes_and_names_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'company_codes_and_names.csv')
    daily_info_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'data_for_models.csv')

    # Load company codes and names
    company_codes = pd.read_csv(company_codes_and_names_path)

    # Find company ID based on code
    company_info = company_codes[company_codes['Company_Code'] == company_code]
    if company_info.empty:
        raise ValueError(f"Company code {company_code} not found in company_codes_and_names.csv")
    company_id = company_info['Company_ID'].values[0]

    # Load stock data
    data = pd.read_csv(daily_info_path)

    # Filter data for the specific company
    data = data[data['company_id'] == company_id]

    # Step 1: Preprocessing
    df = data[['date', 'average_price', 'volume', 'max_price', 'min_price', 'BEST_profit']].copy()

    # Convert 'date' to datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # Handle missing values if any
    df.ffill(inplace=True)

    # Feature columns (excluding the target)
    feature_columns = ['volume', 'max_price', 'min_price', 'BEST_profit']
    target_column = 'average_price'

    # Normalize the features (ensure no outliers cause issues)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[feature_columns].values)

    # Target column
    target = df[target_column].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42)

    # Reshape the data for LSTM (for LSTM input, we need 3D data)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Shape: [samples, time steps (1), features]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])  # Shape: [samples, time steps (1), features]

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),  # [time steps, features]
        LSTM(50, return_sequences=True),
        Dropout(0.4),
        LSTM(50, return_sequences=False),
        Dropout(0.4),
        Dense(25),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 6: Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, verbose=1)

    # Step 7: Make predictions on the test set
    predictions = model.predict(X_test)

    # Step 8: Reverse scaling to get actual prices (from normalized values)
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_features.shape[1] - 1))], axis=1))[:, 0]
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_features.shape[1] - 1))], axis=1))[:, 0]

    # Step 9: Predict the next 7 days' prices
    last_sequence = scaled_features[-1:]  # Take the last observation for prediction
    future_predictions = []

    for _ in range(7):
        pred = model.predict(last_sequence.reshape(1, 1, scaled_features.shape[1]))
        pred_value = pred[0, 0]

        # Clip prediction to the range of recent values (optional)
        min_value = np.min(target[-7:])  # Use recent values for clipping
        max_value = np.max(target[-7:])
        pred_value = np.clip(pred_value, min_value, max_value)

        future_predictions.append(pred_value)

        # Update the sequence with the new prediction
        last_sequence = np.append(last_sequence[0, 1:], [[pred_value] + list(last_sequence[0, 1:])], axis=0).reshape(1, -1)

    # Rescale the future predictions back to the original scale
    future_predictions_rescaled = scaler.inverse_transform(np.concatenate(
        [np.array(future_predictions).reshape(-1, 1), np.zeros((len(future_predictions), scaled_features.shape[1] - 1))],
        axis=1))[:, 0]

    # Step 10: Prepare data for plotting
    dates = df['date'].tolist()[-len(y_test_rescaled):]  # Use the last few dates for the test set
    prices = y_test_rescaled.tolist()

    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 8)]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, label='Actual Prices', color='blue')
    ax.plot(future_dates, future_predictions_rescaled, '--', label='Predicted Prices', color='black')

    for i, (d, p) in enumerate(zip(future_dates, future_predictions_rescaled.flatten())):
        ax.scatter(d, p, color='green' if p > prices[-1] else 'red', s=60)

    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.set_title('Stock Price Prediction for ' + company_code)
    ax.legend()
    ax.grid()

    # Step 11: Save the plot to BytesIO and encode it for use in the web page
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Return base64-encoded image
    return f"data:image/png;base64,{encoded_image}"
