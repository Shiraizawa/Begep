from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

def polynomial_regression_model(train_df, test_df, degree=4):
    # Separate features (X) and target variable (y) for training
    X_train = train_df[['Flight Distance', 'Departure Delay in Minutes']]
    y_train = train_df['Arrival Delay in Minutes']

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)

    # Create a Linear Regression model
    linear_reg_model = LinearRegression()

    # Measure the runtime for model training
    start_time = time.time()

    # Train the model on polynomial features
    linear_reg_model.fit(X_train_poly, y_train)

    # Calculate the training time
    training_time = time.time() - start_time

    # Transform test data to polynomial features
    X_test_poly = poly_features.transform(test_df[['Flight Distance', 'Departure Delay in Minutes']])

    # Make predictions on the test data
    predictions = linear_reg_model.predict(X_test_poly)

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
