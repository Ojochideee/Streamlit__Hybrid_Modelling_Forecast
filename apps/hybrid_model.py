import os
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'apps'))
from data import load_ireland_data, load_uk_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow messages

def apply_fourier_transform_to_all_features(df):
    columns_to_drop = ['Inflation rate']
    existing_columns = df.columns
    columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

    features_to_transform = df.drop(columns=columns_to_drop)

    fft_magnitudes = []
    for col in features_to_transform.columns:
        frequencies = np.fft.fftfreq(len(features_to_transform[col]), d=1)  # Weekly frequency
        fft_result = np.fft.fft(features_to_transform[col])  # Apply FFT to each feature
        positive_freqs = frequencies[frequencies > 0]  # Keep positive frequencies
        fft_magnitude = np.abs(fft_result)  # Magnitude of frequencies
        fft_magnitudes.append(fft_magnitude)
    feature_matrix = np.column_stack(fft_magnitudes)
    return feature_matrix, positive_freqs

def spectral_pca(feature_matrix, n_components=3):
    """Performs Spectral PCA on a feature matrix."""
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(feature_matrix)
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    return pca_scores, explained_variance_ratio, components

def prepare_data_for_modeling(df, n_components=3, test_size=0.3):
    df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime
    df = df.set_index('Date') 
    total_size = len(df)
    test_len = int(total_size * test_size)
    train_len = total_size - test_len
    features = df.drop(columns=['Inflation rate'])
    target = df['Inflation rate']

    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

    #Fourier Transform and PCA
    feature_matrix, _ = apply_fourier_transform_to_all_features(scaled_features_df) # Apply to scaled features
    pca_scores, _, _ = spectral_pca(feature_matrix, n_components=n_components)
    pca_df = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(pca_scores.shape[1])], index=df.index)
    pca_df['Inflation rate'] = target

    #Train-Test-Split
    X_train = pca_df.iloc[:train_len].drop(columns=['Inflation rate'])
    X_test = pca_df.iloc[train_len:].drop(columns=['Inflation rate'])
    y_train = pca_df.iloc[:train_len]['Inflation rate']
    y_test = pca_df.iloc[train_len:]['Inflation rate']
    return X_train, X_test, y_train, y_test


def prepare_lstm_data(X_train, X_test, y_train, y_test, timesteps=12):
    # Reshape into 3D arrays (samples, timesteps, features)
    X_train_lstm = np.array([X_train[i - timesteps:i] for i in range(timesteps, len(X_train))])
    y_train_lstm = y_train[timesteps:]
    X_test_lstm = np.array([X_test[i - timesteps:i] for i in range(timesteps, len(X_test))])
    y_test_lstm = y_test[timesteps:]
    return X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm

def arima_split(df, co_name, test_size=0.3, inflation_column='Inflation rate', exog_cols=['House price index', 'Immigration']):
    y = df[inflation_column]  # Target variable
    exog = df[exog_cols]  # Exogenous variables
    
    # Get the total length of the data
    total_size = len(df)
    test_len = int(total_size * test_size)
    train_len = total_size - test_len
    y_train, y_test = y.iloc[:train_len], y.iloc[train_len:]
    exog_train, exog_test = exog.iloc[:train_len], exog.iloc[train_len:]
    return y_train, y_test, exog_train, exog_test

def app():
    st.markdown('''
        # **Hybrid 4 Model (SARIMAX + LSTM)**

        Welcome to the **Hybrid Model** section of our Inflation Prediction App! On this page, we will showcase the implementation and performance of our hybrid model, which combines the strengths of **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors) and **LSTM** (Long Short-Term Memory networks) to deliver accurate inflation rate predictions.

        ### Key Features of the Hybrid Model:
        - **SARIMAX**: Captures linear trends and seasonal patterns in the time series data.
        - **LSTM**: Handles non-linear dependencies and complex sequences, enhancing prediction accuracy.
        
        The model leverages the unique characteristics of both approaches to address the intricate dynamics of inflation.
        ''')

    #---------------------------------#
    # Sidebar for parameter selection
    st.sidebar.header('Model Parameters')

    # Dataset selection
    # Sidebar - selects country's dataset
    dataset_choice = st.sidebar.selectbox('Choose a Dataset', ['Ireland', 'United Kingdom'])
    if dataset_choice == 'Ireland':
        df = load_ireland_data()
        st.write('**Loaded Ireland Dataset**:')
    else:
        df = load_uk_data()
        st.write('**Loaded United Kingdom Dataset:**')

    # SARIMAX parameters
    st.sidebar.subheader('SARIMAX Parameters')
    sarimax_order = (
        st.sidebar.number_input('SARIMAX Order (p)', min_value=0, value=1),
        st.sidebar.number_input('SARIMAX Order (d)', min_value=0, value=1),
        st.sidebar.number_input('SARIMAX Order (q)', min_value=0, value=1)
    )
    sarimax_seasonal_order = (
        st.sidebar.number_input('SARIMAX Seasonal Order (P)', min_value=0, value=1),
        st.sidebar.number_input('SARIMAX Seasonal Order (D)', min_value=0, value=1),
        st.sidebar.number_input('SARIMAX Seasonal Order (Q)', min_value=0, value=1),
        st.sidebar.number_input('SARIMAX Seasonal Order (s)', min_value=1, value=52)
    )

    # LSTM parameters
    st.sidebar.subheader('LSTM Parameters')
    lstm_units = st.sidebar.number_input('LSTM Units', min_value=1, value=50)
    lstm_epochs = st.sidebar.number_input('LSTM Epochs', min_value=1, value=50)
    lstm_batch_size = st.sidebar.number_input('LSTM Batch Size', min_value=1, value=32)
    lstm_timesteps = st.sidebar.number_input('LSTM Timesteps', min_value=1, value=12)



    #---------------------------------#
    # Main panel
    
    # SARIMAX Model
    st.header('SARIMAX Model')
    target_column = 'Inflation rate'  # Specify target column for SARIMAX
    exog_columns = ['House price index', 'Immigration']

    # Save and print exogenous variable names
    st.write("Exogenous Variables:", exog_columns)

    y_train_sarimax, y_test_sarimax, exog_train, exog_test = arima_split(df, dataset_choice,inflation_column=target_column,exog_cols=exog_columns)
    # Print debug information
    st.write("SARIMAX Train Shape:", y_train_sarimax.shape)
    st.write("SARIMAX Test Shape:", y_test_sarimax.shape)

    sarimax = SARIMAX(y_train_sarimax, exog=exog_train, order=sarimax_order, seasonal_order=sarimax_seasonal_order)
    sarimax_fit = sarimax.fit(disp=False)
    sarimax_train_preds = sarimax_fit.predict(start=0, end=len(y_train_sarimax) - 1, exog=exog_train)
    sarimax_test_preds = sarimax_fit.predict(start=len(y_train_sarimax), end=len(y_train_sarimax) + len(y_test_sarimax) - 1, exog=exog_test)

    # Evaluation metrics for SARIMAX
    sarimax_mae = mean_absolute_error(y_test_sarimax, sarimax_test_preds)
    sarimax_mse = mean_squared_error(y_test_sarimax, sarimax_test_preds)
    sarimax_rmse = np.sqrt(sarimax_mse)
    sarimax_mape = np.mean(np.abs((y_test_sarimax - sarimax_test_preds) / y_test_sarimax)) * 100

    st.write('SARIMAX Evaluation Metrics:')
    st.write('MAE:', sarimax_mae)
    st.write('MSE:', sarimax_mse)
    st.write('RMSE:', sarimax_rmse)
    st.write('MAPE:', sarimax_mape)
    


    #---------------------------------#
    # LSTM Model
    # Prepare data for LSTM
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = prepare_lstm_data(X_train, X_test, y_train, y_test, lstm_timesteps)
    
    st.header('LSTM Model')
    st.write("LSTM Train Shape:", X_train_lstm.shape)
    st.write("LSTM Test Shape:", X_test_lstm.shape)
    lstm_model = Sequential([
        LSTM(lstm_units, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=1)
    lstm_test_pred = lstm_model.predict(X_test_lstm).flatten()

    # Evaluation metrics for LSTM
    lstm_mae = mean_absolute_error(y_test_lstm, lstm_test_pred)
    lstm_mse = mean_squared_error(y_test_lstm, lstm_test_pred)
    lstm_rmse = np.sqrt(lstm_mse)
    lstm_mape = np.mean(np.abs((y_test_lstm - lstm_test_pred) / y_test_lstm)) * 100

    st.write('LSTM Evaluation Metrics:')
    st.write('MAE:', lstm_mae)
    st.write('MSE:', lstm_mse)
    st.write('RMSE:', lstm_rmse)
    st.write('MAPE:', lstm_mape)



    #---------------------------------#
    st.header('HYBRID 4 Model')
    # Convert SARIMAX predictions to NumPy arrays and reshape
    sarimax_train_preds = sarimax_train_preds.to_numpy().reshape(-1, 1)
    sarimax_test_preds = sarimax_test_preds.to_numpy().reshape(-1, 1)

    # Calculate residuals from SARIMAX predictions
    residuals_train = y_train_sarimax.to_numpy().reshape(-1, 1) - sarimax_train_preds

    # Extract residuals for LSTM
    residuals_lstm_train = residuals_train[:len(y_train_lstm)]

    # Train LSTM on SARIMAX residuals
    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_units, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, residuals_lstm_train, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=1)
    lstm_test_predict = lstm_model.predict(X_test_lstm).flatten()

    # Reshape LSTM predictions
    lstm_test_predict = lstm_test_predict.reshape(-1, 1)

    # Trim SARIMAX and y_test to match the length of LSTM predictions
    sarimax_test_preds_trimmed = sarimax_test_preds[-len(lstm_test_predict):]
    y_test_trimmed = y_test_sarimax[-len(lstm_test_predict):].to_numpy().reshape(-1, 1)

    # Combine SARIMAX and LSTM predictions
    hybrid_test_preds = sarimax_test_preds_trimmed + lstm_test_predict

    # Flatten for evaluation
    sarimax_test_preds_trimmed = sarimax_test_preds_trimmed.flatten()
    y_test_trimmed = y_test_trimmed.flatten()
    hybrid_test_preds = hybrid_test_preds.flatten()

    # Evaluation metrics for Hybrid Model
    hybrid_mae = mean_absolute_error(y_test_trimmed, hybrid_test_preds)
    hybrid_mse = mean_squared_error(y_test_trimmed, hybrid_test_preds)
    hybrid_rmse = np.sqrt(hybrid_mse)
    hybrid_mape = np.mean(np.abs((y_test_trimmed - hybrid_test_preds) / y_test_trimmed)) * 100

    st.write('Hybrid Evaluation Metrics:')
    st.write('MAE:', hybrid_mae)
    st.write('MSE:', hybrid_mse)
    st.write('RMSE:', hybrid_rmse)
    st.write('MAPE:', hybrid_mape)

    # Ensure predictions and actual values have the same length
    y_test_trimmed = y_test_sarimax[-len(hybrid_test_preds):].to_numpy()
    sarimax_test_preds_trimmed = sarimax_test_preds[-len(hybrid_test_preds):]
    lstm_test_preds_trimmed = lstm_test_predict.flatten()



    #---------------------------------#
    # Create the plot for Model Predictions vs Actual Values

    st.header('Model Predictions vs Actual Values')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_trimmed, label='Actual Values', color='blue', linewidth=2)
    ax.plot(sarimax_test_preds_trimmed, label='SARIMAX Predictions', color='green', linestyle='--', linewidth=2)
    ax.plot(lstm_test_preds_trimmed, label='LSTM Predictions', color='red', linestyle='-.', linewidth=2)
    ax.plot(hybrid_test_preds, label='Hybrid Model Predictions', color='orange', linestyle='-', linewidth=2)

    # Add labels, title, legend, and grid
    ax.set_title('Actual vs SARIMAX, LSTM, and Hybrid Model Predictions', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Inflation Rate', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


    #---------------------------------#
    # Future Predictions Section
    st.header('Future Predictions')
    forecast_steps = 104

    # SARIMAX Forecast for future values
    sarimax_forecast = sarimax_fit.predict(start=len(y_train_sarimax), end=len(y_train_sarimax) + forecast_steps - 1, exog=exog_test[:forecast_steps])

    # Create residuals from the SARIMAX forecast (we will use the last value of the residuals)
    sarimax_train_preds = sarimax_fit.predict(start=0, end=len(y_train_sarimax) - 1, exog=exog_train)
    # Ensure that sarimax_train_preds and y_train_sarimax are numpy arrays
    sarimax_train_preds = sarimax_train_preds.to_numpy().reshape(-1, 1)  # If it's a pandas Series
    y_train_sarimax_values = y_train_sarimax.to_numpy().reshape(-1, 1)

    # Compute residuals
    residuals_train = y_train_sarimax_values - sarimax_train_preds

    # Prepare LSTM input for future predictions
    future_lstm_input = X_test_lstm[-1].reshape(1, X_test_lstm.shape[1], X_test_lstm.shape[2])
    future_lstm_preds = []
    for _ in range(forecast_steps):
        future_pred = lstm_model.predict(future_lstm_input)
        future_lstm_preds.append(future_pred.flatten()[0])
        future_lstm_input = np.roll(future_lstm_input, -1, axis=1)
        future_lstm_input[0, -1, 0] = future_pred

    future_lstm_preds = np.array(future_lstm_preds)

    # Combine SARIMAX and LSTM residuals to generate Hybrid predictions
    hybrid_future_preds = sarimax_forecast + future_lstm_preds

    # Historical plot data
    y_test_trimmed = y_test_sarimax[-len(hybrid_test_preds):].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_trimmed, label='Actual Values', color='blue', linewidth=2)
    ax.plot(sarimax_test_preds_trimmed, label='SARIMAX Predictions', color='green', linestyle='--', linewidth=2)
    ax.plot(lstm_test_preds_trimmed, label='LSTM Predictions', color='red', linestyle='-.', linewidth=2)
    ax.plot(hybrid_test_preds, label='Hybrid Model Predictions', color='orange', linestyle='-', linewidth=2)
    ax.plot(range(len(y_test_trimmed), len(y_test_trimmed) + forecast_steps), hybrid_future_preds, label='Hybrid Future Predictions', color='purple', linestyle='-', linewidth=2)
    ax.set_title('Hybrid Model Future predictions', fontsize=16)
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Inflation Rate', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
