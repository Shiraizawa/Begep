from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

def neural_network_regression_model(train_df, test_df, epochs=25, batch_size=64, validation_split=0.3):
    # Separate features (X) and target variable (y) for training
    X_train = train_df[['Flight Distance', 'Departure Delay in Minutes']]
    y_train = train_df['Arrival Delay in Minutes']

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_df[['Flight Distance', 'Departure Delay in Minutes']])

    # Create a Neural Network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')


    # Measure the runtime for model training
    start_time = time.time()

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)

    # Calculate the training time
    training_time = time.time() - start_time

    # Make predictions on the test data
    predictions = model.predict(X_test_scaled).flatten()

    # Evaluate the model
    actual_values = test_df['Arrival Delay in Minutes']

    # Mean Squared Error
    mse = mean_squared_error(actual_values, predictions)

    # Mean Absolute Error
    mae = mean_absolute_error(actual_values, predictions)

    # R-squared
    r_squared = r2_score(actual_values, predictions)

    # Scatter plot for actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(test_df['Arrival Delay in Minutes'], predictions, color='blue', label='Predictions')
    plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # Return the metrics and the plot
    return mse, mae, r_squared, training_time, plt
