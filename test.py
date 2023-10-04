import streamlit as st
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



air_temp = np.array([ 27.64,  30.80,  33.61,  34.84,  37.45,  32.55,  32.41,  31.38,  31.71,  32.48,  29.75,  29.09,  29.18,  32.40,  34.62,  37.34,  37.13,  33.53,  30.98,  31.39,  30.79,  31.76,  30.09,  28.77,  28.68,  31.68,  33.94,  36.47,  37.67,  33.08,  30.77,  31.28,  32.34,  31.70,  29.66,  28.43,  28.52,  31.93,  34.66,  35.66,  35.90,  34.25,  30.31,  30.29,  31.76,  32.06,  31.14,  27.23,  27.86,  31.09,  33.84,  36.33,  37.17,  34.95,  31.23,  30.59,  30.32,  30.93,  30.35,  28.18,  28.21,  29.38,  33.13,  35.06,  35.99,  32.90,  31.45,  29.92,  31.76,  30.87,  29.69,  28.54,  29.44,  30.84,  34.71,  35.81,  35.78,  32.78,  30.82,  31.38,  30.70,  31.70,  28.96,  27.15,  27.09,  29.92,  34.40,  36.04,  35.91,  33.99,  30.34,  30.64,  31.20,  30.71,  29.60,  28.94])
water_temp = np.array([26, 26, 26, 26, 28, 27, 27, 25, 25, 27, 26, 26, 29, 29, 29, 28, 28, 28, 27, 26, 26, 29, 27, 28, 27, 27, 27, 29, 29, 31, 30, 31, 31, 27, 26, 26, 27, 30, 30, 30, 30, 28, 26, 27, 27, 27, 25, 25, 26, 28, 30, 29, 36, 29, 26, 27, 24, 25, 27, 28, 27, 26, 26, 26, 30, 29, 28, 26, 26, 26, 26, 25, 23, 26, 26, 32, 32, 32, 32, 26, 30, 30, 27, 28, 28, 25, 25, 27, 27, 25, 25, 25, 25, 25, 25, 25])
air_temp = air_temp.reshape(-1, 1)
water_temp = water_temp.reshape(-1, 1)



def calculate_saturated_dissolved_oxygen(water_temp):
    water_temp = water_temp
    if(water_temp == 0): 
        return -1 
    return math.exp(- 139.34 + ((1.575701 * 10**5) / water_temp) - 
            ((6.642308 * 10**7) / (water_temp**2)) + 
            ((1.2438 * 10**10) / (water_temp**3)) - 
            ((8.621949 * 10**11) / (water_temp**4)))

lr_model = LinearRegression()
lr_model.fit(air_temp, water_temp)

svr_model = SVR(kernel='sigmoid')
svr_model.fit(air_temp, water_temp.flatten())

decision_tree_model = DecisionTreeRegressor()
decision_tree_model.fit(air_temp, water_temp)

gradient_boosting_model = GradientBoostingRegressor()
gradient_boosting_model.fit(air_temp, water_temp.flatten())

rf_model = RandomForestRegressor()
rf_model.fit(air_temp, water_temp.flatten())

ridge_model = Ridge()
ridge_model.fit(air_temp, water_temp)

def predict_water_temperature(air_temp, model):
    prediction = model.predict(np.array(air_temp).reshape(-1, 1))
    return prediction

st.title("River Water Quality Prediction Tool")

task_selection = st.radio("Select a Task", ["River Water Temperature Prediction", "Saturated Dissolved Oxygen"])
if task_selection == "River Water Temperature Prediction":
    st.header("River Water Temperature Prediction")
    st.write("Use the options below to predict river water temperature.")

    air_temp_selected = st.number_input("Enter Air Temperature", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    observed_water_temp_selected = st.number_input("Enter Observed Water Temperature", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ml_model_selected = st.selectbox("Select Machine Learning Model", ["Linear Regression", "Random Forest", "Ridge", "Support Vector Regression", "Decision Tree", "Gradient Boosting"])
    performance_metrics_selected = st.multiselect("Select Performance Metrics", ["RMSE", "MAE", "MSE", "R2"])

    if st.button("Submit"):
        if ml_model_selected == "Linear Regression":
            prediction = predict_water_temperature(air_temp_selected, lr_model)
        elif ml_model_selected == "Random Forest":
            prediction = predict_water_temperature(air_temp_selected, rf_model)
        elif ml_model_selected == "Ridge":
            prediction = predict_water_temperature(air_temp_selected, ridge_model)
        elif ml_model_selected == "Support Vector Regression":
            prediction = predict_water_temperature(air_temp_selected, svr_model)
        elif ml_model_selected == "Decision Tree":
            prediction = predict_water_temperature(air_temp_selected, decision_tree_model)
        elif ml_model_selected == "Gradient Boosting":
            prediction = predict_water_temperature(air_temp_selected, gradient_boosting_model)

        mse = mean_squared_error([observed_water_temp_selected], [prediction])
        mae = mean_absolute_error([observed_water_temp_selected], [prediction])
        r2 = r2_score([observed_water_temp_selected], [prediction])
        rmse = np.sqrt(mse)

        st.write("Predicted Water Temperature:", prediction[0])
        st.write("Mean Squared Error (MSE):", mse)
        st.write("Mean Absolute Error (MAE):", mae)
        st.write("Root Mean Squared Error (RMSE):", rmse)
        st.write("R2 Score:", r2)

        results_df = pd.DataFrame({
            "Air Temperature": [air_temp_selected],
            "Observed Water Temperature": [observed_water_temp_selected],
            "Predicted Water Temperature": [prediction[0]],
            "RMSE": [rmse],
            "MAE": [mae],
            "MSE": [mse],
            "R2": [r2]
        })

        st.write("Download Output")
        st.write(results_df)
        csv_export_button = st.button("Export as CSV")
        if csv_export_button:
            st.write("Downloading CSV...")
            results_df.to_csv("water_temperature_prediction_results.csv", index=False)
            st.success("CSV file saved successfully!")

elif task_selection == "Saturated Dissolved Oxygen":
    st.header("Saturated Dissolved Oxygen Module")
    st.write("Select the source for river water temperature:")

    source_option = st.radio("Select Source", ["Select Data", "Simulate Data"])

    if source_option == "Select Data":
        st.header("Saturated Dissolved Oxygen Module")
        st.write("Please upload a file containing the water temperature for a specific year:")
        uploaded_file = st.file_uploader("Choose a file", type=["txt"])

        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            water_temperature = float(content.strip()) + 273.15
            
            st.write("Water Temperature:", water_temperature)
            saturated_oxygen = calculate_saturated_dissolved_oxygen(water_temperature)
            st.write("Saturated Dissolved Oxygen:", (saturated_oxygen))

    elif source_option == "Simulate Data":
        st.write("Simulating Data...")
        observed_water_temp_selected = st.selectbox("Select Observed Water Temperature Data", water_temp.flatten())

        next_water_temperature = observed_water_temp_selected + 1  
        next_water_temperature = next_water_temperature + 273.00
        simulated_saturated_oxygen = calculate_saturated_dissolved_oxygen(next_water_temperature)

        st.write("Next Simulated Water Temperature in Kelvin:")
        st.write(next_water_temperature )
        st.write("Simulated Saturated Dissolved Oxygen for the Next Water Temperature:")
        st.write(math.exp(simulated_saturated_oxygen))