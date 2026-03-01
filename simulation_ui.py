import itertools

import numpy as np
import pandas as pd
import streamlit as st

from app import TripInput, _predict_emissions, calculate_fees, encoder, model_load_error


st.set_page_config(
    page_title="Nairobi Taxi Emissions Simulator",
    layout="wide",
)


DEFAULT_VEHICLES = ["Aqua", "Axio", "Demio", "Leaf", "Note", "Prius", "Vitz"]
DEFAULT_FUELS = ["Diesel", "EV", "Gasoline", "Hybrid"]


def inject_green_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f3fbf5 0%, #e9f7ee 45%, #e2f3ea 100%);
            color: #0f3d2e;
        }
        .block-container {
            padding-top: 1.2rem;
        }
        .stSidebar {
            background: #e2f3e9;
        }
        .stButton > button, .stDownloadButton > button {
            background: #2f9e44;
            color: white;
            border: none;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #237a34;
            color: white;
        }
        .eco-banner {
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 12px;
            background: linear-gradient(90deg, #1b5e20 0%, #2e7d32 55%, #388e3c 100%);
            color: #f5fff7;
            box-shadow: 0 8px 18px rgba(26, 94, 32, 0.2);
        }
        .icon-card {
            border: 1px solid #cdebd6;
            background: #f7fff9;
            border-radius: 12px;
            padding: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            min-height: 72px;
        }
        .icon-card-title {
            font-size: 0.88rem;
            color: #1b5e20;
            margin-bottom: 2px;
        }
        .icon-card-value {
            font-size: 0.95rem;
            font-weight: 600;
            color: #12452f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _svg_icon(kind: str) -> str:
    icons = {
        "leaf": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M20 4C13 4 6 7.5 6 14c0 3 2 6 5 6 6.5 0 9-7 9-16Z" '
            'stroke="#2e7d32" stroke-width="1.8"/><path d="M8 19c2.5-3.5 5.5-6 9-8" stroke="#2e7d32" '
            'stroke-width="1.8" stroke-linecap="round"/></svg>'
        ),
        "chart": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M4 19h16M7 16v-4M12 16V7M17 16v-7" '
            'stroke="#1b5e20" stroke-width="1.8" stroke-linecap="round"/></svg>'
        ),
        "policy": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M12 3 4 7v6c0 5 3.5 7.8 8 8 4.5-.2 8-3 8-8V7l-8-4Z" '
            'stroke="#1b5e20" stroke-width="1.8"/><path d="m9 12 2 2 4-4" stroke="#1b5e20" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        ),
    }
    return icons.get(kind, icons["leaf"])


def icon_card(kind: str, title: str, value: str):
    st.markdown(
        f"""
        <div class="icon-card">
            <div>{_svg_icon(kind)}</div>
            <div>
                <div class="icon-card-title">{title}</div>
                <div class="icon-card-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _available_categories():
    if encoder is not None and hasattr(encoder, "categories_"):
        vehicle_types = [str(x) for x in encoder.categories_[0]]
        fuel_types = [str(x) for x in encoder.categories_[1]]
        return vehicle_types, fuel_types
    return DEFAULT_VEHICLES, DEFAULT_FUELS


def simulate_trip(
    trip_id: int,
    distance_km: float,
    vehicle_type: str,
    fuel_type: str,
    base_fee: float,
    emission_cost_per_kg: float,
) -> dict:
    trip = TripInput(
        trip_id=trip_id,
        distance_km=distance_km,
        vehicle_type=vehicle_type,
        fuel_type=fuel_type,
    )
    predicted_emissions = _predict_emissions(trip)
    fee = calculate_fees(
        emissions=predicted_emissions,
        base_fee=base_fee,
        emission_cost_per_kg=emission_cost_per_kg,
    )
    emissions_per_km = predicted_emissions / distance_km if distance_km > 0 else 0.0
    return {
        "trip_id": trip_id,
        "distance_km": round(distance_km, 2),
        "vehicle_type": vehicle_type,
        "fuel_type": fuel_type,
        "predicted_emissions_kg": round(predicted_emissions, 4),
        "calculated_fee": round(fee, 2),
        "emissions_per_km": round(emissions_per_km, 4),
    }


def build_scenario_dataframe(
    distances: list[float],
    vehicle_types: list[str],
    fuel_types: list[str],
    base_fee: float,
    emission_cost_per_kg: float,
) -> pd.DataFrame:
    rows = []
    trip_id = 1000
    for distance_km, vehicle, fuel in itertools.product(distances, vehicle_types, fuel_types):
        rows.append(
            simulate_trip(
                trip_id=trip_id,
                distance_km=distance_km,
                vehicle_type=vehicle,
                fuel_type=fuel,
                base_fee=base_fee,
                emission_cost_per_kg=emission_cost_per_kg,
            )
        )
        trip_id += 1
    return pd.DataFrame(rows)


def app_header():
    st.markdown(
        """
        <div class="eco-banner">
            <h2 style="margin:0 0 4px 0;">Nairobi Taxi Emissions & Fee Simulation Studio</h2>
            <div style="opacity:0.95;">
                Run detailed what-if simulations across trip distance, vehicle type, fuel type,
                and pricing policy to support cleaner transport decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Run detailed what-if simulations across trip distance, vehicle type, fuel type, and "
        "pricing policy to understand emissions and fee impacts."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        icon_card("leaf", "Sustainability lens", "Track CO2 outcomes by scenario")
    with c2:
        icon_card("chart", "Data-driven simulation", "Compare outcomes across many trips")
    with c3:
        icon_card("policy", "Policy prototyping", "Tune fee rules and target thresholds")


def sidebar_controls(vehicle_choices: list[str], fuel_choices: list[str]):
    st.sidebar.header("Simulation Controls")

    st.sidebar.subheader("Fee Model")
    base_fee = st.sidebar.number_input("Base fee", min_value=0.0, value=5.0, step=0.5)
    emission_cost = st.sidebar.number_input(
        "Emission cost per kg CO2",
        min_value=0.0,
        value=2.0,
        step=0.1,
    )

    st.sidebar.subheader("Policy Targets")
    target_emissions = st.sidebar.number_input(
        "Target emissions per trip (kg CO2)",
        min_value=0.0,
        value=1.5,
        step=0.1,
    )
    target_fee = st.sidebar.number_input(
        "Target fee per trip",
        min_value=0.0,
        value=8.0,
        step=0.1,
    )

    st.sidebar.subheader("Scenario Scope")
    selected_vehicles = st.sidebar.multiselect(
        "Vehicle types to include",
        options=vehicle_choices,
        default=vehicle_choices,
    )
    selected_fuels = st.sidebar.multiselect(
        "Fuel types to include",
        options=fuel_choices,
        default=fuel_choices,
    )

    return {
        "base_fee": base_fee,
        "emission_cost": emission_cost,
        "target_emissions": target_emissions,
        "target_fee": target_fee,
        "selected_vehicles": selected_vehicles,
        "selected_fuels": selected_fuels,
    }


def single_trip_panel(
    vehicle_choices: list[str],
    fuel_choices: list[str],
    base_fee: float,
    emission_cost: float,
    target_emissions: float,
    target_fee: float,
):
    st.subheader("Single Trip Deep-Dive")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        trip_id = st.number_input("Trip ID", min_value=1, value=1, step=1)
    with col2:
        distance_km = st.slider("Distance (km)", min_value=1.0, max_value=100.0, value=12.0, step=0.5)
    with col3:
        vehicle = st.selectbox("Vehicle type", options=vehicle_choices, index=0)
    with col4:
        fuel = st.selectbox("Fuel type", options=fuel_choices, index=min(2, len(fuel_choices) - 1))

    if st.button("Run Single Trip Simulation", type="primary", use_container_width=True):
        try:
            result = simulate_trip(
                trip_id=int(trip_id),
                distance_km=float(distance_km),
                vehicle_type=vehicle,
                fuel_type=fuel,
                base_fee=base_fee,
                emission_cost_per_kg=emission_cost,
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Predicted emissions", f"{result['predicted_emissions_kg']} kg CO2")
        metric2.metric("Calculated fee", f"{result['calculated_fee']}")
        metric3.metric("Emissions intensity", f"{result['emissions_per_km']} kg/km")
        budget_status = "Within target" if result["calculated_fee"] <= target_fee else "Above target"
        metric4.metric("Fee target status", budget_status)

        ev_compare = None
        if fuel != "EV" and "EV" in fuel_choices:
            try:
                ev_compare = simulate_trip(
                    trip_id=int(trip_id),
                    distance_km=float(distance_km),
                    vehicle_type=vehicle,
                    fuel_type="EV",
                    base_fee=base_fee,
                    emission_cost_per_kg=emission_cost,
                )
            except Exception:
                ev_compare = None

        detail_col1, detail_col2 = st.columns([1.3, 1])
        with detail_col1:
            st.markdown("#### Simulation Output")
            st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)
        with detail_col2:
            st.markdown("#### Target Gauges")
            emissions_ratio = min(result["predicted_emissions_kg"] / max(target_emissions, 0.001), 1.0)
            fee_ratio = min(result["calculated_fee"] / max(target_fee, 0.001), 1.0)
            st.write("Emissions vs target")
            st.progress(emissions_ratio)
            st.write("Fee vs target")
            st.progress(fee_ratio)

            if ev_compare:
                savings = result["predicted_emissions_kg"] - ev_compare["predicted_emissions_kg"]
                if savings > 0:
                    st.success(f"Switching to EV could save ~{savings:.2f} kg CO2 for this trip.")
                else:
                    st.info("EV comparison does not reduce emissions for this specific combination.")


def scenario_panel(
    selected_vehicles: list[str],
    selected_fuels: list[str],
    base_fee: float,
    emission_cost: float,
):
    st.subheader("Batch Scenario Simulation")
    st.caption("Generate multiple trips and compare outcomes across many combinations.")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_distance = st.number_input("Min distance (km)", min_value=1.0, value=5.0, step=1.0)
    with c2:
        max_distance = st.number_input("Max distance (km)", min_value=1.0, value=40.0, step=1.0)
    with c3:
        steps = st.number_input("Distance points", min_value=2, value=8, step=1)

    if not selected_vehicles or not selected_fuels:
        st.warning("Select at least one vehicle type and one fuel type in the sidebar.")
        return

    if min_distance > max_distance:
        st.error("Minimum distance must be less than or equal to maximum distance.")
        return

    distances = [float(x) for x in pd.Series(np.linspace(min_distance, max_distance, int(steps))).round(2).tolist()]

    if st.button("Run Batch Scenarios", use_container_width=True):
        try:
            df = build_scenario_dataframe(
                distances=distances,
                vehicle_types=selected_vehicles,
                fuel_types=selected_fuels,
                base_fee=base_fee,
                emission_cost_per_kg=emission_cost,
            )
        except Exception as exc:
            st.error(f"Batch simulation failed: {exc}")
            return

        if df.empty:
            st.warning("No scenarios generated.")
            return

        st.markdown("#### Scenario Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scenarios", f"{len(df)}")
        k2.metric("Avg emissions", f"{df['predicted_emissions_kg'].mean():.3f} kg")
        k3.metric("Avg fee", f"{df['calculated_fee'].mean():.2f}")
        k4.metric("Best low-emission scenario", df.loc[df["predicted_emissions_kg"].idxmin(), "fuel_type"])

        st.markdown("#### Detailed Results")
        st.dataframe(
            df.sort_values(["predicted_emissions_kg", "calculated_fee"]),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="simulation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### Emissions by Fuel Type")
            fuel_emissions = (
                df.groupby("fuel_type", as_index=False)["predicted_emissions_kg"]
                .mean()
                .sort_values("predicted_emissions_kg")
            )
            st.bar_chart(fuel_emissions.set_index("fuel_type"))

        with chart_col2:
            st.markdown("#### Fee by Vehicle Type")
            vehicle_fee = (
                df.groupby("vehicle_type", as_index=False)["calculated_fee"]
                .mean()
                .sort_values("calculated_fee")
            )
            st.bar_chart(vehicle_fee.set_index("vehicle_type"))

        st.markdown("#### Distance Sensitivity (Emissions vs Distance)")
        line_df = (
            df.groupby(["distance_km", "fuel_type"], as_index=False)["predicted_emissions_kg"]
            .mean()
            .pivot(index="distance_km", columns="fuel_type", values="predicted_emissions_kg")
            .sort_index()
        )
        st.line_chart(line_df)

        st.markdown("#### Top 10 Lowest-Emission Scenarios")
        top10 = df.nsmallest(10, "predicted_emissions_kg")
        st.dataframe(top10, use_container_width=True, hide_index=True)


def main():
    inject_green_theme()
    app_header()

    if model_load_error is not None:
        st.error(
            "Model is not ready. Install required dependencies and ensure model files exist.\n\n"
            f"Details: {model_load_error}"
        )
        st.stop()

    vehicle_choices, fuel_choices = _available_categories()
    controls = sidebar_controls(vehicle_choices, fuel_choices)

    single_trip_panel(
        vehicle_choices=vehicle_choices,
        fuel_choices=fuel_choices,
        base_fee=controls["base_fee"],
        emission_cost=controls["emission_cost"],
        target_emissions=controls["target_emissions"],
        target_fee=controls["target_fee"],
    )

    st.divider()
    scenario_panel(
        selected_vehicles=controls["selected_vehicles"],
        selected_fuels=controls["selected_fuels"],
        base_fee=controls["base_fee"],
        emission_cost=controls["emission_cost"],
    )


if __name__ == "__main__":
    main()
