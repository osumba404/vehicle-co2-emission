# Nairobi Taxi Emissions and Fee Simulator

This project predicts taxi trip emissions and calculates trip fees using a trained machine learning model.

It includes:

- A **FastAPI backend** in `app.py`
- A **Streamlit simulation dashboard** in `simulation_ui.py`
- Pretrained model artifacts:
  - `co2_model.pkl`
  - `encoder.pkl`

The dashboard is designed for rich simulation workflows: single-trip analysis, batch scenario generation, visual comparisons, and CSV export.

## Features

- Emissions prediction for user-defined trip inputs
- Fee calculation based on configurable policy parameters
- EV comparison suggestion for non-EV trips
- Detailed scenario simulation across distances, vehicle types, and fuel types
- Data tables + charts for:
  - Emissions by fuel type
  - Fee by vehicle type
  - Emissions sensitivity vs distance
- Downloadable CSV of simulation outputs
- Green-themed UI styled for sustainability-focused analysis

## Project Structure

```text
infinityai_hack/
├─ app.py
├─ simulation_ui.py
├─ requirements.txt
├─ co2_model.pkl
├─ encoder.pkl
└─ nairobi_taxi_synthetic.csv
```

## Requirements

- Python 3.10+ (3.14 works)
- pip

Install all dependencies from `requirements.txt`.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Run the Streamlit Simulation UI (Recommended)

```powershell
python -m streamlit run simulation_ui.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

### What you can do in the UI

- Configure policy values (`base_fee`, `emission_cost_per_kg`)
- Set emissions and fee targets
- Simulate a single trip in detail
- Run batch simulations across many combinations
- View charts and ranked scenario tables
- Export scenario results as CSV

## Run the FastAPI Backend

```powershell
python app.py
```

API docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoint

### `POST /predict_emissions_and_fees`

Request body:

```json
{
  "trip_id": 1,
  "distance_km": 12.5,
  "vehicle_type": "Axio",
  "fuel_type": "Gasoline"
}
```

Response shape:

```json
{
  "predicted_emissions_kg": 1.857,
  "calculated_fee": 8.71,
  "recommendation": "Choosing an EV for a similar trip could save approximately 0.42 kg of CO2."
}
```

## Supported Categories

Based on the loaded encoder, typical values include:

- Vehicle types: `Aqua`, `Axio`, `Demio`, `Leaf`, `Note`, `Prius`, `Vitz`
- Fuel types: `Diesel`, `EV`, `Gasoline`, `Hybrid`

## Troubleshooting

### 1) `No module named streamlit`

Cause: You installed `streamlit` outside your active virtual environment.

Fix:

```powershell
.\.venv\Scripts\python -m pip install streamlit
```

### 2) `Model is not ready ... No module named 'xgboost'`

Cause: The serialized model depends on `xgboost`.

Fix:

```powershell
.\.venv\Scripts\python -m pip install xgboost
```

### 3) Model or encoder file not found

Ensure these files exist in the project root:

- `co2_model.pkl`
- `encoder.pkl`

### 4) Version mismatch warnings (scikit-learn / xgboost)

You may see warnings when loading older serialized models with newer package versions. The app can still run, but for production-grade reliability retraining and re-exporting the model with your current toolchain is recommended.

## Development Notes

- Core prediction preprocessing aligns input features to `model.feature_names_in_` when available.
- The UI imports prediction logic directly from `app.py` to keep behavior consistent across API and dashboard.
- For reproducibility, always run commands through the project `.venv`.

## Next Enhancements (Optional)

- Add monthly fleet simulation and policy stress-testing tabs
- Add map or route segmentation inputs
- Persist simulation history to file/database
- Add automated tests for API and simulation calculations
