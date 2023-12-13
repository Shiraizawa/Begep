from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

def knn_model(train_df, test_df, n_neighbors=9):
    # Separate features (X) and target variable (y) for training
    X_train = train_df[['Flight Distance', 'Departure Delay in Minutes']]
    y_train = train_df['Arrival Delay in Minutes']

    # Create a K-Nearest Neighbors Regression model
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)

    # Measure the runtime for model training
    start_time = time.time()

    # Train the KNN model
    knn_model.fit(X_train, y_train)

    # Calculate the training time
    training_time = time.time() - start_time
    

    # Make predictions on the test data
    X_test = test_df[['Flight Distance', 'Departure Delay in Minutes']]
    predictions = knn_model.predict(X_test)

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
    plt.scatter(actual_values, predictions, color='blue', label='Predictions')
    plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # Return the metrics and the plot
    return mse, mae, r_squared,training_time, plt

