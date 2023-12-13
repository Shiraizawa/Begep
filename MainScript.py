import pandas as pd
import streamlit as st
from KNeighborsRegressionModel import knn_model
from LinearRegressionModel import linear_model
from BayesianRidgeRegressionModel import bayesian_ridge_model
from DecisionTreeRegressionModel import decision_tree_model
from ElasticNetRegressionModel import elastic_net_model
from LassoRegressionModel import lasso_model
from NeuralNetworkRegressionModel import neural_network_regression_model
from PolynomialRegression import polynomial_regression_model
from RandomForestRegressionModel import random_forest_regression_model
from RidgeRegressionModel import ridge_regression_model
from SVRegressionModel import svr_regression_model

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


# Load data
train_csv_path = 'train.csv'
test_csv_path = 'test.csv'

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

selected_columns = ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
train_df = train_df[selected_columns].dropna()
test_df = test_df[selected_columns].dropna()

# Define models
models = {
    'K-Nearest Neighbors': knn_model,
    'Linear Regression': linear_model,
    'Bayesian Ridge Regression': bayesian_ridge_model,
    'Decision Tree Regression': decision_tree_model,
    'Elastic Net Regression': elastic_net_model,
    'Lasso Regression': lasso_model,
    'Neural Network Regression': neural_network_regression_model,
    'Polynomial Regression': polynomial_regression_model,
    'Random Forest Regression': random_forest_regression_model,
    'Ridge Regression': ridge_regression_model,
    'Support Vector Regression': svr_regression_model
}

# Streamlit app
st.title("Regression Model Evaluation App")

# Model selection
selected_model = st.radio("Select a Regression Model", list(models.keys()))

# Button to trigger model evaluation
if st.button("Evaluate Model"):
    st.write(f"### {selected_model} Evaluation Results:")

    # Call the selected model function
    model_function = models[selected_model]
    mse, mae, r_squared, training_time, model_plot = model_function(train_df, test_df)

    st.write(f"Training Time: {training_time:.2f} seconds")
    st.write(f"R-squared: {r_squared}")
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Mean Squared Error: {mse}")

    # Display the plot
    st.pyplot(model_plot)

    st.success("Model evaluation completed!")

