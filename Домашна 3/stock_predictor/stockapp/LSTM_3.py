import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for image generation
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
import io
import base64
from matplotlib.figure import Figure
import os
from django.conf import settings

def predict_stock_prices(company_code):
    company_codes_and_names_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'company_codes_and_names.csv')
    daily_info_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'data_for_models.csv')

    company_and_names = pd.read_csv(company_codes_and_names_path)

    # Find company ID based on code
    company_info = company_and_names[company_and_names['Company_Code'] == company_code]
    if company_info.empty:
        raise ValueError(f"Company code {company_code} not found in company_codes_and_names.csv")
    company_id = company_info['Company_ID'].values[0]

    df = pd.read_csv(daily_info_path)
    # Filter data for the specific company
    df = df[df['company_id'] == company_id]

    #Covnert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # ascending=False
    df = df.sort_values(by='date')

    # Potrebni koloni
    df = df[['date', 'average_price', 'volume', 'max_price', 'min_price']].copy()

    # Handle missing values if any
    df.ffill(inplace=True)
    df.bfill(inplace=True)


    # Check if we have enough data for LSTM (at least 60 records)
    def check_if_last_and_30th_are_similar(df, threshold=0.01):
        last_record = df.iloc[-1]
        record_30th = df.iloc[-31]

        columns_to_check = ['max_price', 'min_price', 'average_price', 'volume']  # Adjust columns as needed

        differences = abs(last_record[columns_to_check] - record_30th[columns_to_check])

        if all(differences < threshold):
            return True
        else:
            return False

    # Assuming is_similar is determined by your check function
    is_similar = check_if_last_and_30th_are_similar(df)

    if len(df)<60 or is_similar:
        # If there are less than 60 records, predict the last value repeated 7 times
        last_row = df.iloc[-1]
        next_7_days = pd.date_range(last_row['date'] + timedelta(days=1), periods=7, freq='D')

        # Repeat the last known values for the next 7 days
        next_7_days_data = {
            'date': next_7_days,
            'average_price': [last_row['average_price']] * 7,
            'volume': [last_row['volume']] * 7,
            'max_price': [last_row['max_price']] * 7,
            'min_price': [last_row['min_price']] * 7
        }

        next_7_days_df = pd.DataFrame(next_7_days_data)

        # Get the last 30 days of actual data
        actual_dates_last_30 = df['date'].iloc[-30:].values  # Assuming df is your main data DataFrame
        y_test_last_30 = df['average_price'].iloc[-30:].values  # Assuming 'average_price' is the column you want

        # Plotting the last 30 days and the predicted next 7 days
        plt.figure(figsize=(12, 6))

        # Plot actual prices for the last 30 days
        plt.plot(actual_dates_last_30, y_test_last_30, label='Вистинска цена (Последни 30 денови)', color='#8368E8')

        # Add the last actual date and value to the predicted data
        next_dates_with_connection = [actual_dates_last_30[-1]] + list(next_7_days_df['date'])  # Add last actual date
        next_7_days_predictions_with_connection = [y_test_last_30[-1]] + list(
            next_7_days_df['average_price'])  # Add last actual price

        # Plot the predicted prices for the next 7 days
        plt.plot(next_dates_with_connection, next_7_days_predictions_with_connection, label="Предвидени цени (Наредни 7 денови)", color='#8368E8', linestyle='--')

        # Add plot labels and title
        plt.xlabel('Датум', labelpad=20)
        plt.ylabel('Цена на акција (денари)', labelpad=20)
        plt.title(f'Недоволно/Невалидни податоци за тренирање на LSTM модел за компанијата {company_code}', color='red')
        plt.legend()
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Set the background color of the entire figure (outside the plot area)
        plt.gcf().set_facecolor('#f9f9f9')  # Color outside the plot area (the canvas)

        # Save the plot to BytesIO and encode it for use in the web page
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Return base64-encoded image
        return f"data:image/png;base64,{encoded_image}"

    # Min and Max date - for scaling
    min_date = pd.to_datetime(df['date'].min())
    max_date = pd.to_datetime(df['date'].max())

    # Scale the dates to a range between 0 and 1
    df['scaled_date'] = (df['date'] - min_date) / (max_date - min_date)

    # Get features and target
    X, y = df.drop(['average_price', 'date'], axis=1), df['average_price']

    # Split into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    # Apply MinMaxScaler to scale the features (fit on training set, transform both train and test sets)
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)  # Fit and transform on training set
    X_test_scaled = x_scaler.transform(X_test)

    # Reshape the data for LSTM input format: [samples, time steps, features]
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1,
                                            X_train_scaled.shape[1])  # [samples, time steps, features]
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1,
                                          X_test_scaled.shape[1])  # [samples, time steps, features]
    # Apply MinMaxScaler to scale the target variable (fit on training set, transform both train and test sets)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))  # Fit and transform on training set
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    model = Sequential([
        Input((X_train_scaled.shape[1], X_train_scaled.shape[2],)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Step 6: Train the model
    model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled), epochs=20, batch_size=64, verbose=1, callbacks=[early_stopping])

    # Step 7: Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test_scaled, y_test_scaled)
    print(f"Test Loss (MSE): {test_loss}")

    # Alternatively, calculate Mean Squared Error (MSE) manually
    mse = mean_squared_error(y_test_scaled, predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    # Inverse transform the predictions to get them back to the original price range
    predictions_original = y_scaler.inverse_transform(predictions)

    # Access scaled dates from the first column of X_test
    scaled_dates = X_test.iloc[:, -1]  # Access the last column (scaled_date)

    # Convert the scaled dates back to actual dates
    actual_dates = min_date + pd.to_timedelta(scaled_dates * (max_date - min_date), unit='D')

    # Step 1: Get the last 7 days from the original data
    last_7_days = df.tail(7)
    last_date = df['date'].iloc[-1]

    # Calculate the next 7 days by extrapolating based on the recent trend
    price_diff_max = (last_7_days['max_price'].iloc[-1] - last_7_days['max_price'].iloc[0]) / len(last_7_days)
    price_diff_min = (last_7_days['min_price'].iloc[-1] - last_7_days['min_price'].iloc[0]) / len(last_7_days)
    price_diff_avg = (last_7_days['average_price'].iloc[-1] - last_7_days['average_price'].iloc[0]) / len(last_7_days)
    volume_diff = (last_7_days['volume'].iloc[-1] - last_7_days['volume'].iloc[0]) / len(last_7_days)

    next_7_days_max = [last_7_days['max_price'].iloc[-1] + price_diff_max * (i + 1) for i in range(7)]
    next_7_days_min = [last_7_days['min_price'].iloc[-1] + price_diff_min * (i + 1) for i in range(7)]
    next_7_days_avg = [last_7_days['average_price'].iloc[-1] + price_diff_avg * (i + 1) for i in range(7)]
    next_7_days_volume = [last_7_days['volume'].iloc[-1] + volume_diff * (i + 1) for i in range(7)]

    # Generate the dates for the next 7 days
    next_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq='D')

    # Scale the next dates
    next_dates_scaled = (next_dates - min_date).days / (max_date - min_date).days

    # Create the DataFrame for the next 7 days
    next_7_days_df = pd.DataFrame({
        'volume': next_7_days_volume,
        'max_price': next_7_days_max,
        'min_price': next_7_days_min,
        'scaled_date': next_dates_scaled
    })

    # Step 2: Scale the new data (X) using the x_scaler
    next_7_days_scaled = x_scaler.transform(next_7_days_df[['volume', 'max_price', 'min_price', 'scaled_date']])

    # Step 3: Reshape the scaled data for the model
    next_7_days_features_reshaped = next_7_days_scaled.reshape(
        (next_7_days_scaled.shape[0], 1, next_7_days_scaled.shape[1])
    )

    # Step 4: Make predictions using the model
    next_7_days_predictions_scaled = model.predict(next_7_days_features_reshaped)

    # Step 5: If y was scaled, inverse transform the predictions
    next_7_days_predictions_original = y_scaler.inverse_transform(next_7_days_predictions_scaled)

    # Prepare the actual dates for plotting
    actual_dates = min_date + pd.to_timedelta(X_test['scaled_date'] * (max_date - min_date).days, unit='D')



    """
    # Limit the displayed data to the last 30 days of the test set
    last_30_indices = -30 if len(actual_dates) > 30 else -len(actual_dates)
    actual_dates_last_30 = actual_dates[last_30_indices:]
    predictions_original_last_30 = predictions_original[last_30_indices:]
    y_test_last_30 = y_test.values[last_30_indices:]
    
    # Ensure next_dates and next_7_days_predictions_original are 1D arrays
    next_dates = np.array(next_dates).flatten()
    next_7_days_predictions_original = np.array(next_7_days_predictions_original).flatten()
    
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot the last 30 days of test data predictions with reduced opacity (alpha=0.25)
    plt.plot(actual_dates_last_30, predictions_original_last_30, label='Предвидени цени (Последни 30 денови)', color='red', alpha=0.25)

    # Connect the first dot of next 7 days' predictions with the last actual price of the last 30 days
    # Ensure the next_dates and next_7_days_predictions_original are 1D and have compatible shapes
    next_dates_with_connection = [actual_dates_last_30.iloc[-1]] + list(
        next_dates)  # Add the last date of last 30 days to next_dates
    next_7_days_predictions_with_connection = [y_test_last_30[-1]] + list(
        next_7_days_predictions_original)  # Add the last actual price to next predictions

    # Plot next 7 days' predictions with the first dot connected to the actual price
    plt.plot(next_dates_with_connection, next_7_days_predictions_with_connection, label="Предвидени цени (Наредни 7 денови)", color='green', linestyle='--')

    # Plot actual prices for the last 30 days
    plt.plot(actual_dates_last_30, y_test_last_30, label='Вистинска цена (Последни 30 денови)', color='green')

    # Add plot labels and title
    plt.xlabel('Датум')
    plt.ylabel('Цена на акција')
    plt.title(f'LSTM Предвидување на цена на акциите: Последни 30 и наредни 7 денови за {company_code}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.figtext(0.5, 0.01, 'Имајте во предвид дека информациите за наредните денови се автоматски генерирани и не се вистинити!', ha='center', fontsize=10, color='red')
    """

    # Ensure next_dates and next_7_days_predictions_original are 1D arrays
    next_dates = np.array(next_dates).flatten()
    next_7_days_predictions_original = np.array(next_7_days_predictions_original).flatten()

    # Get the last 30 days of actual data
    actual_dates_last_30 = df['date'].iloc[-30:].values  # Assuming df is your main data DataFrame
    y_test_last_30 = df['average_price'].iloc[-30:].values  # Assuming 'average_price' is the column you want

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot actual prices for the last 30 days
    plt.plot(actual_dates_last_30, y_test_last_30, label='Вистинска цена (Последни 30 денови)', color='#8368E8')

    # Add the last actual date and value to the predicted data
    next_dates_with_connection = [actual_dates_last_30[-1]] + list(next_dates)  # Add last actual date
    next_7_days_predictions_with_connection = [y_test_last_30[-1]] + list(
        next_7_days_predictions_original)  # Add last actual price

    # Plot the predicted prices for the next 7 days
    plt.plot(next_dates_with_connection, next_7_days_predictions_with_connection,
             label="Предвидени цени (Наредни 7 денови)", color='#8368E8', linestyle='--')

    # Add labels and title
    plt.xlabel('Датум')
    plt.ylabel('Цена на акција (денари)')
    plt.title(f'LSTM Предвидување на цена на акциите за {company_code}')

    # Set the background color of the entire figure (outside the plot area)
    plt.gcf().set_facecolor('#f9f9f9')  # Color outside the plot area (the canvas)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)

    # Add the legend
    plt.legend()

    # Tight layout to prevent clipping
    plt.tight_layout()


    # Show the plot
    #plt.show()

    # Save the plot to BytesIO and encode it for use in the web page
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Return base64-encoded image
    return f"data:image/png;base64,{encoded_image}"
