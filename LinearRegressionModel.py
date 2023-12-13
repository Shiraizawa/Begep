import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import time

def linear_model(train_df, test_df):
    # Separate features (X) and target variable (y) for training
    X_train = train_df[['Flight Distance', 'Departure Delay in Minutes']]
    y_train = train_df['Arrival Delay in Minutes']

    # Create a Linear Regression model
    model = LinearRegression()

    # Measure the runtime for model training
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Calculate the training time
    training_time = time.time() - start_time
    

    # Make predictions on the test data
    X_test = test_df[['Flight Distance', 'Departure Delay in Minutes']]
    predictions = model.predict(X_test)

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

    return mse, mae, r_squared, training_time, plt

# Example usage
train_csv_path = 'train.csv'
test_csv_path = 'test.csv'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

selected_columns = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
train_df = train_df[selected_columns].dropna()
test_df = test_df[selected_columns].dropna()

mse, mae, r_squared, training_time, plt = linear_model(train_df, test_df)
print(f'\nTraining Time: {training_time:.2f} seconds')
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r_squared}')

plt.show()

# Now you can use mse, mae, and r_squared in your further analysis or display them as needed.
