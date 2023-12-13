from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

def elastic_net_model(train_df, test_df, alpha=0.8, l1_ratio=0.7):
    # Separate features (X) and target variable (y) for training
    X_train = train_df[['Flight Distance', 'Departure Delay in Minutes']]
    y_train = train_df['Arrival Delay in Minutes']

    # Create an Elastic Net Regression model
    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    # Measure the runtime for model training
    start_time = time.time()

    # Train the Elastic Net model
    elastic_net_model.fit(X_train, y_train)

    # Calculate the training time
    training_time = time.time() - start_time

    # Make predictions on the test data
    X_test = test_df[['Flight Distance', 'Departure Delay in Minutes']]
    predictions = elastic_net_model.predict(X_test)

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
