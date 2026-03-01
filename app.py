# This code block is a conceptual representation of app.py
# It is not executable within the Colab environment as it requires a web server framework (FastAPI)
# and is meant to be saved as a Python file (e.g., app.py) and run in a separate environment.

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Load trained model + encoder
try:
    model = joblib.load("co2_model.pkl")
    encoder = joblib.load("encoder.pkl")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model or encoder file not found. Ensure 'co2_model.pkl' and 'encoder.pkl' are in the same directory.")

# FastAPI app
app = FastAPI(title="Nairobi Taxi Emissions and Fees API", version="1.0")

# Request schema based on the expected input features after one-hot encoding
# We need to define a schema that can handle the raw input before encoding,
# then perform the encoding within the prediction function.
class TripInput(BaseModel):
    trip_id: int # Although not used in prediction, it's in the original data
    distance_km: float
    vehicle_type: str # Categorical
    fuel_type: str # Categorical

# Define the calculate_fees function within the API code
def calculate_fees(emissions, base_fee=5, emission_cost_per_kg=2):
  """
  Calculates fees based on predicted emissions.

  Args:
    emissions: Predicted emissions in kg.
    base_fee: The base fee for a trip.
    emission_cost_per_kg: The cost per kg of emissions.

  Returns:
    The calculated fee.
  """
  return base_fee + emissions * emission_cost_per_kg


@app.post("/predict_emissions_and_fees")
def predict_emissions_and_fees(trip: TripInput):
    # Convert input to dataframe
    # Create a dictionary with the input data
    input_data = {
        'trip_id': trip.trip_id,
        'distance_km': trip.distance_km,
        'vehicle_type': trip.vehicle_type,
        'fuel_type': trip.fuel_type
    }
    # Convert the dictionary to a pandas DataFrame
    df_input = pd.DataFrame([input_data])

    # --- Preprocessing steps to match the training data format ---

    # Identify categorical columns used during training
    categorical_cols_trained = ['vehicle_type', 'fuel_type'] # Based on original df columns

    # Apply the loaded OneHotEncoder to the categorical columns
    try:
        # Use the loaded encoder to transform the input data
        df_encoded_input = encoder.transform(df_input[categorical_cols_trained])
        # Create a DataFrame from the encoded output, using the feature names from the encoder
        df_encoded_input = pd.DataFrame(df_encoded_input, columns=encoder.get_feature_names_out(categorical_cols_trained))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during encoding: {e}")


    # Recreate the structure of X_train, ensuring all columns are present and in the correct order
    # This requires knowing the exact column names and order from the training data (X_train)
    # A robust API would save the list of training columns during the training phase and load it here.
    # For this example, we'll assume the column order and names are known or can be derived.
    # A better approach for production would be to save X_train.columns.tolist() during training.

    # Let's reconstruct the expected columns based on the original df_encoded columns
    # from the Colab notebook state.
    # This is a simplified approach; in production, save the training columns.
    expected_columns = [col for col in df_encoded.columns if col != 'emissions_kg'] # Use df_encoded from Colab state

    # Initialize a new DataFrame with the expected columns, filled with zeros/False
    df_processed_input = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Populate the new DataFrame with the input data
    df_processed_input['trip_id'] = df_input['trip_id'] # Assuming trip_id is not dropped
    df_processed_input['distance_km'] = df_input['distance_km']

    # Populate the one-hot encoded columns
    for col in df_encoded_input.columns:
        if col in df_processed_input.columns:
            df_processed_input[col] = df_encoded_input[col]

    # Ensure the columns are in the exact order as X_train
    # This is crucial for the model prediction
    # A production system would load the saved X_train column order.
    # For this example, we will use the columns from X_train in the Colab state
    expected_column_order = X_train.columns.tolist() # Using X_train from Colab state

    # Reindex the processed input DataFrame to match the training column order
    try:
         df_processed_input = df_processed_input.reindex(columns=expected_column_order, fill_value=0)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error reindexing input columns: {e}")


    # --- End of Preprocessing ---

    # Make prediction
    try:
        predicted_emissions = model.predict(df_processed_input)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")


    # Calculate fees
    calculated_fee = calculate_fees(predicted_emissions)

    # Prepare recommendation (simple example)
    # This would ideally be more sophisticated, perhaps comparing to average for the same distance etc.
    recommendation = None
    if trip.fuel_type != 'EV':
        # Estimate savings compared to the average gasoline car emissions (using the average calculated earlier)
        # This requires having access to the average_gasoline_emissions value.
        # In a production API, this average should be saved and loaded, or calculated periodically.
        # For this example, we'll use the value from the Colab notebook state.
        # Ensure average_gasoline_emissions is available in the API environment.
        try:
             estimated_savings = average_gasoline_emissions - predicted_emissions
             if estimated_savings > 0.1: # Only suggest if savings are significant
                recommendation = f"Choosing an EV for a similar trip could save approximately {estimated_savings:.2f} kg of CO₂."
        except NameError:
             # Handle case where average_gasoline_emissions is not available
             pass # Or log a warning


    return {
        "predicted_emissions_kg": round(float(predicted_emissions), 4),
        "calculated_fee": round(float(calculated_fee), 2),
        "recommendation": recommendation
    }

# To run this API locally for testing:
# 1. Save the code as a Python file (e.g., app.py).
# 2. Make sure you have FastAPI and uvicorn installed (`pip install fastapi uvicorn[standard]`).
# 3. Ensure 'co2_model.pkl' and 'encoder.pkl' are in the same directory.
# 4. Run the command: `uvicorn app:app --reload`
# 5. Access the API documentation at http://127.0.0.1:8000/docs