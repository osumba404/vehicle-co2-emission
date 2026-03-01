# This code block is a conceptual representation of app.py
# It is not executable within the Colab environment as it requires a web server framework (FastAPI)
# and is meant to be saved as a Python file (e.g., app.py) and run in a separate environment.

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# FastAPI app
app = FastAPI(title="Nairobi Taxi Emissions and Fees API", version="1.0")

# Load trained model + encoder once at startup.
model = None
encoder = None
model_feature_names = []
model_load_error = None

try:
    model = joblib.load("co2_model.pkl")
    encoder = joblib.load("encoder.pkl")
    model_feature_names = [str(col) for col in getattr(model, "feature_names_in_", [])]
except Exception as exc:
    model_load_error = str(exc)

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


def _ensure_model_ready():
    if model_load_error is not None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model failed to load. Ensure 'co2_model.pkl' and 'encoder.pkl' are present "
                f"and dependencies are installed (e.g., xgboost). Details: {model_load_error}"
            ),
        )


def _build_model_input(trip: TripInput) -> pd.DataFrame:
    _ensure_model_ready()

    df_input = pd.DataFrame(
        [{
            "trip_id": trip.trip_id,
            "distance_km": trip.distance_km,
            "vehicle_type": trip.vehicle_type,
            "fuel_type": trip.fuel_type,
        }]
    )

    categorical_cols = ["vehicle_type", "fuel_type"]

    try:
        encoded = encoder.transform(df_input[categorical_cols])
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        df_encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df_input.index,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error during encoding: {exc}")

    df_features = pd.concat([df_input[["trip_id", "distance_km"]], df_encoded], axis=1)

    # Align exactly with the training feature order when available.
    if model_feature_names:
        df_features = df_features.reindex(columns=model_feature_names, fill_value=0)

    return df_features


def _predict_emissions(trip: TripInput) -> float:
    df_model_input = _build_model_input(trip)
    try:
        return float(model.predict(df_model_input)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {exc}")


@app.post("/predict_emissions_and_fees")
def predict_emissions_and_fees(trip: TripInput):
    predicted_emissions = _predict_emissions(trip)

    # Calculate fees
    calculated_fee = calculate_fees(predicted_emissions)

    # Compare with an equivalent EV trip for a practical recommendation.
    recommendation = None
    if trip.fuel_type != "EV":
        try:
            ev_trip = TripInput(
                trip_id=trip.trip_id,
                distance_km=trip.distance_km,
                vehicle_type=trip.vehicle_type,
                fuel_type="EV",
            )
            ev_emissions = _predict_emissions(ev_trip)
            estimated_savings = predicted_emissions - ev_emissions
            if estimated_savings > 0.1:
                recommendation = (
                    f"Choosing an EV for a similar trip could save approximately "
                    f"{estimated_savings:.2f} kg of CO2."
                )
        except HTTPException:
            # Keep the core response usable even if comparison prediction fails.
            recommendation = None

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)