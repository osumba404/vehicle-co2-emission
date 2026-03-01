import itertools

import numpy as np
import pandas as pd
import streamlit as st

from app import TripInput, _predict_emissions, calculate_fees, encoder, model_load_error


st.set_page_config(
    page_title="Nairobi Taxi Emissions Simulator",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DEFAULT_VEHICLES = ["Aqua", "Axio", "Demio", "Leaf", "Note", "Prius", "Vitz"]
DEFAULT_FUELS = ["Diesel", "EV", "Gasoline", "Hybrid"]


def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(165deg, #f4fbf6 0%, #eff8f2 45%, #e8f4ec 100%);
            color: #173c2a;
        }
        [data-testid="stSidebar"] {
            background: #edf7f1;
        }
        [data-testid="stHeader"] {
            background: rgba(244, 251, 246, 0.85);
        }
        .block-container {
            padding-top: 1.1rem;
            max-width: 1300px;
        }

        /* Force readable label text across widgets */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stNumberInput label,
        .stTextInput label {
            color: #173c2a !important;
            font-weight: 500;
        }

        .hero {
            background: linear-gradient(110deg, #1f6a3d 0%, #2d7a4b 60%, #358454 100%);
            color: #f6fff8;
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 12px 28px rgba(29, 96, 60, 0.22);
            margin-bottom: 14px;
        }
        .hero h2 {
            margin: 0 0 6px 0;
            font-size: 1.55rem;
            color: #f6fff8;
        }
        .hero p {
            margin: 0;
            opacity: 0.96;
            font-size: 1rem;
            color: #f6fff8;
        }
        .card {
            background: #f9fffb;
            border: 1px solid #d2eadb;
            border-radius: 12px;
            padding: 12px 14px;
            min-height: 78px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .card-title {
            color: #1f6a3d;
            font-size: 0.86rem;
            margin-bottom: 2px;
        }
        .card-value {
            color: #173c2a;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .section-note {
            background: #f4fbf7;
            border: 1px solid #d7ede0;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
            color: #173c2a;
        }
        .stButton > button, .stDownloadButton > button {
            background: #2f8f54;
            color: #ffffff;
            border: none;
            border-radius: 9px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #237546;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _icon(kind: str) -> str:
    icons = {
        "leaf": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M20 4C13 4 6 7.5 6 14c0 3 2 6 5 6 6.5 0 9-7 9-16Z" '
            'stroke="#2b7a4b" stroke-width="1.8"/><path d="M8 19c2.5-3.5 5.5-6 9-8" stroke="#2b7a4b" '
            'stroke-width="1.8" stroke-linecap="round"/></svg>'
        ),
        "chart": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M4 19h16M7 16v-4M12 16V7M17 16v-7" '
            'stroke="#1f6a3d" stroke-width="1.8" stroke-linecap="round"/></svg>'
        ),
        "policy": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M12 3 4 7v6c0 5 3.5 7.8 8 8 4.5-.2 8-3 8-8V7l-8-4Z" '
            'stroke="#1f6a3d" stroke-width="1.8"/><path d="m9 12 2 2 4-4" stroke="#1f6a3d" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        ),
    }
    return icons.get(kind, icons["leaf"])


def feature_card(kind: str, title: str, value: str):
    st.markdown(
        f"""
        <div class="card">
            <div>{_icon(kind)}</div>
            <div>
                <div class="card-title">{title}</div>
                <div class="card-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_header():
    st.markdown(
        """
        <div class="hero">
            <h2>Nairobi Taxi Emissions & Fee Simulation Studio</h2>
            <p>
                Explore cleaner mobility decisions with easy-to-read controls, scenario comparisons,
                and visual insights for emissions and fare outcomes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("leaf", "Sustainability", "See which choices reduce CO2 per trip")
    with c2:
        feature_card("chart", "Clarity", "Compare many scenarios in one place")
    with c3:
        feature_card("policy", "Policy testing", "Tune fees and measure trade-offs quickly")


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


def common_controls(vehicle_choices: list[str], fuel_choices: list[str]) -> dict:
    with st.expander("Policy settings and scenario scope", expanded=True):
        st.markdown(
            """
            <div class="section-note">
                Use these shared settings for both tabs.
                Labels are written in plain language to make each control easier to understand.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            base_fee = st.number_input(
                "Base trip fee",
                min_value=0.0,
                value=5.0,
                step=0.5,
                help="The fixed fee charged before environmental surcharge is added.",
            )
        with c2:
            emission_cost = st.number_input(
                "Environmental surcharge per kilogram of CO2",
                min_value=0.0,
                value=2.0,
                step=0.1,
                help="Additional cost for each kg of predicted CO2.",
            )
        with c3:
            target_emissions = st.number_input(
                "Target CO2 per trip (kg)",
                min_value=0.0,
                value=1.5,
                step=0.1,
                help="Used to show whether a trip is below or above your emissions target.",
            )
        with c4:
            target_fee = st.number_input(
                "Target trip fee",
                min_value=0.0,
                value=8.0,
                step=0.1,
                help="Used to show whether a trip is within your fee goal.",
            )

        c5, c6 = st.columns(2)
        with c5:
            selected_vehicles = st.multiselect(
                "Vehicle types to include in batch simulation",
                options=vehicle_choices,
                default=vehicle_choices,
            )
        with c6:
            selected_fuels = st.multiselect(
                "Fuel types to include in batch simulation",
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


def _display_columns_config():
    return {
        "trip_id": st.column_config.NumberColumn("Trip ID", format="%d"),
        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
        "vehicle_type": st.column_config.TextColumn("Vehicle"),
        "fuel_type": st.column_config.TextColumn("Fuel"),
        "predicted_emissions_kg": st.column_config.NumberColumn("Predicted CO2 (kg)", format="%.4f"),
        "calculated_fee": st.column_config.NumberColumn("Estimated fee", format="%.2f"),
        "emissions_per_km": st.column_config.NumberColumn("CO2 intensity (kg/km)", format="%.4f"),
    }


def trip_simulator_tab(
    vehicle_choices: list[str],
    fuel_choices: list[str],
    controls: dict,
):
    st.subheader("Trip Simulator")
    st.caption("Simulate one trip and see easy-to-read sustainability and fee indicators.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        trip_id = st.number_input("Trip identifier", min_value=1, value=1, step=1)
    with c2:
        distance_km = st.slider("Trip distance (km)", min_value=1.0, max_value=100.0, value=12.0, step=0.5)
    with c3:
        vehicle = st.selectbox("Vehicle type", options=vehicle_choices)
    with c4:
        fuel = st.selectbox("Fuel type", options=fuel_choices, index=min(2, len(fuel_choices) - 1))

    run_single = st.button("Run trip simulation", type="primary", use_container_width=True)

    if run_single:
        try:
            result = simulate_trip(
                trip_id=int(trip_id),
                distance_km=float(distance_km),
                vehicle_type=vehicle,
                fuel_type=fuel,
                base_fee=controls["base_fee"],
                emission_cost_per_kg=controls["emission_cost"],
            )
            st.session_state["single_trip_result"] = result
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return

    result = st.session_state.get("single_trip_result")
    if not result:
        st.info("Enter trip details and click 'Run trip simulation' to view outputs.")
        return

    emissions_delta = result["predicted_emissions_kg"] - controls["target_emissions"]
    fee_delta = result["calculated_fee"] - controls["target_fee"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Predicted CO2",
        f"{result['predicted_emissions_kg']:.4f} kg",
        delta=f"{emissions_delta:+.3f} vs target",
        delta_color="inverse",
    )
    m2.metric(
        "Estimated trip fee",
        f"{result['calculated_fee']:.2f}",
        delta=f"{fee_delta:+.2f} vs target",
        delta_color="inverse",
    )
    m3.metric("CO2 intensity", f"{result['emissions_per_km']:.4f} kg/km")
    m4.metric("Distance", f"{result['distance_km']:.2f} km")

    ev_comparison_msg = "No EV comparison available."
    if result["fuel_type"] != "EV" and "EV" in fuel_choices:
        try:
            ev_result = simulate_trip(
                trip_id=result["trip_id"],
                distance_km=result["distance_km"],
                vehicle_type=result["vehicle_type"],
                fuel_type="EV",
                base_fee=controls["base_fee"],
                emission_cost_per_kg=controls["emission_cost"],
            )
            savings = result["predicted_emissions_kg"] - ev_result["predicted_emissions_kg"]
            if savings > 0:
                ev_comparison_msg = f"Switching to EV could reduce emissions by approximately {savings:.2f} kg."
            else:
                ev_comparison_msg = "For this input, EV does not reduce emissions in the current model."
        except Exception:
            ev_comparison_msg = "EV comparison could not be computed for this trip."

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("#### Trip output table")
        st.dataframe(
            pd.DataFrame([result]),
            use_container_width=True,
            hide_index=True,
            column_config=_display_columns_config(),
        )
    with right:
        st.markdown("#### Quick interpretation")
        st.write(f"- **CO2 target check:** {'Below target' if emissions_delta <= 0 else 'Above target'}")
        st.write(f"- **Fee target check:** {'Within target' if fee_delta <= 0 else 'Above target'}")
        st.write(f"- **EV comparison:** {ev_comparison_msg}")


def scenario_explorer_tab(controls: dict):
    st.subheader("Scenario Explorer")
    st.caption("Generate many trips at once and compare outcomes with polished tables and charts.")

    if not controls["selected_vehicles"] or not controls["selected_fuels"]:
        st.warning("Choose at least one vehicle type and one fuel type in policy settings.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        min_distance = st.number_input("Minimum distance (km)", min_value=1.0, value=5.0, step=1.0)
    with c2:
        max_distance = st.number_input("Maximum distance (km)", min_value=1.0, value=40.0, step=1.0)
    with c3:
        steps = st.number_input("Number of distance points", min_value=2, value=8, step=1)

    if min_distance > max_distance:
        st.error("Minimum distance must be less than or equal to maximum distance.")
        return

    distances = [float(x) for x in np.linspace(min_distance, max_distance, int(steps)).round(2).tolist()]

    run_batch = st.button("Generate scenario set", use_container_width=True)
    if run_batch:
        try:
            df = build_scenario_dataframe(
                distances=distances,
                vehicle_types=controls["selected_vehicles"],
                fuel_types=controls["selected_fuels"],
                base_fee=controls["base_fee"],
                emission_cost_per_kg=controls["emission_cost"],
            )
            st.session_state["scenario_df"] = df
        except Exception as exc:
            st.error(f"Batch simulation failed: {exc}")
            return

    df = st.session_state.get("scenario_df")
    if df is None or df.empty:
        st.info("Click 'Generate scenario set' to display scenario insights.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scenarios generated", f"{len(df)}")
    k2.metric("Average CO2", f"{df['predicted_emissions_kg'].mean():.3f} kg")
    k3.metric("Average fee", f"{df['calculated_fee'].mean():.2f}")
    k4.metric("Lowest-CO2 fuel (average)", df.groupby("fuel_type")["predicted_emissions_kg"].mean().idxmin())

    st.markdown("#### Scenario results table")
    results_sorted = df.sort_values(["predicted_emissions_kg", "calculated_fee"]).reset_index(drop=True)
    st.dataframe(
        results_sorted,
        use_container_width=True,
        hide_index=True,
        column_config=_display_columns_config(),
    )

    st.download_button(
        "Download scenario results as CSV",
        data=results_sorted.to_csv(index=False).encode("utf-8"),
        file_name="scenario_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("#### Average CO2 by fuel type")
        fuel_emissions = (
            df.groupby("fuel_type", as_index=False)["predicted_emissions_kg"]
            .mean()
            .sort_values("predicted_emissions_kg")
            .set_index("fuel_type")
        )
        st.bar_chart(fuel_emissions)

    with chart2:
        st.markdown("#### Average fee by vehicle type")
        vehicle_fee = (
            df.groupby("vehicle_type", as_index=False)["calculated_fee"]
            .mean()
            .sort_values("calculated_fee")
            .set_index("vehicle_type")
        )
        st.bar_chart(vehicle_fee)

    st.markdown("#### CO2 trend as distance increases")
    trend_df = (
        df.groupby(["distance_km", "fuel_type"], as_index=False)["predicted_emissions_kg"]
        .mean()
        .pivot(index="distance_km", columns="fuel_type", values="predicted_emissions_kg")
        .sort_index()
    )
    st.line_chart(trend_df)

    st.markdown("#### Top 10 lowest-emission scenarios")
    st.dataframe(
        df.nsmallest(10, "predicted_emissions_kg").reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config=_display_columns_config(),
    )


def guide_tab():
    st.subheader("How to use this dashboard")
    st.markdown(
        """
        1. Open **Policy settings and scenario scope** and set your fee policy and targets.  
        2. Use **Trip Simulator** for one specific trip.  
        3. Use **Scenario Explorer** to compare many combinations and export results.  
        4. Watch the target deltas to quickly see whether scenarios are above or below your goals.
        """
    )


def main():
    inject_theme()
    app_header()

    if model_load_error is not None:
        st.error(
            "Model is not ready. Install required dependencies and ensure model files exist.\n\n"
            f"Details: {model_load_error}"
        )
        st.stop()

    vehicle_choices, fuel_choices = _available_categories()
    controls = common_controls(vehicle_choices, fuel_choices)

    tab1, tab2, tab3 = st.tabs(["Trip Simulator", "Scenario Explorer", "Guide"])
    with tab1:
        trip_simulator_tab(vehicle_choices, fuel_choices, controls)
    with tab2:
        scenario_explorer_tab(controls)
    with tab3:
        guide_tab()


if __name__ == "__main__":
    main()
import itertools

import numpy as np
import pandas as pd
import streamlit as st

from app import TripInput, _predict_emissions, calculate_fees, encoder, model_load_error


st.set_page_config(
    page_title="Nairobi Taxi Emissions Simulator",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DEFAULT_VEHICLES = ["Aqua", "Axio", "Demio", "Leaf", "Note", "Prius", "Vitz"]
DEFAULT_FUELS = ["Diesel", "EV", "Gasoline", "Hybrid"]


def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(165deg, #f4fbf6 0%, #eff8f2 45%, #e8f4ec 100%);
            color: #173c2a;
        }
        [data-testid="stSidebar"] {
            background: #edf7f1;
        }
        [data-testid="stHeader"] {
            background: rgba(244, 251, 246, 0.85);
        }
        .block-container {
            padding-top: 1.1rem;
            max-width: 1300px;
        }
        .hero {
            background: linear-gradient(110deg, #1f6a3d 0%, #2d7a4b 60%, #358454 100%);
            color: #f6fff8;
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 12px 28px rgba(29, 96, 60, 0.22);
            margin-bottom: 14px;
        }
        .hero h2 {
            margin: 0 0 6px 0;
            font-size: 1.55rem;
        }
        .hero p {
            margin: 0;
            opacity: 0.96;
            font-size: 1rem;
        }
        .card {
            background: #f9fffb;
            border: 1px solid #d2eadb;
            border-radius: 12px;
            padding: 12px 14px;
            min-height: 78px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .card-title {
            color: #1f6a3d;
            font-size: 0.86rem;
            margin-bottom: 2px;
        }
        .card-value {
            color: #173c2a;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .section-note {
            background: #f4fbf7;
            border: 1px solid #d7ede0;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        .stButton > button, .stDownloadButton > button {
            background: #2f8f54;
            color: #ffffff;
            border: none;
            border-radius: 9px;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #237546;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _icon(kind: str) -> str:
    icons = {
        "leaf": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M20 4C13 4 6 7.5 6 14c0 3 2 6 5 6 6.5 0 9-7 9-16Z" '
            'stroke="#2b7a4b" stroke-width="1.8"/><path d="M8 19c2.5-3.5 5.5-6 9-8" stroke="#2b7a4b" '
            'stroke-width="1.8" stroke-linecap="round"/></svg>'
        ),
        "chart": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M4 19h16M7 16v-4M12 16V7M17 16v-7" '
            'stroke="#1f6a3d" stroke-width="1.8" stroke-linecap="round"/></svg>'
        ),
        "policy": (
            '<svg width="28" height="28" viewBox="0 0 24 24" fill="none" '
            'xmlns="http://www.w3.org/2000/svg"><path d="M12 3 4 7v6c0 5 3.5 7.8 8 8 4.5-.2 8-3 8-8V7l-8-4Z" '
            'stroke="#1f6a3d" stroke-width="1.8"/><path d="m9 12 2 2 4-4" stroke="#1f6a3d" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        ),
    }
    return icons.get(kind, icons["leaf"])


def feature_card(kind: str, title: str, value: str):
    st.markdown(
        f"""
        <div class="card">
            <div>{_icon(kind)}</div>
            <div>
                <div class="card-title">{title}</div>
                <div class="card-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_header():
    st.markdown(
        """
        <div class="hero">
            <h2>Nairobi Taxi Emissions & Fee Simulation Studio</h2>
            <p>
                Explore cleaner mobility decisions with easy-to-read controls, scenario comparisons,
                and visual insights for emissions and fare outcomes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("leaf", "Sustainability", "See which choices reduce CO2 per trip")
    with c2:
        feature_card("chart", "Clarity", "Compare many scenarios in one place")
    with c3:
        feature_card("policy", "Policy testing", "Tune fees and measure trade-offs quickly")


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


def common_controls(vehicle_choices: list[str], fuel_choices: list[str]) -> dict:
    with st.expander("Policy settings and scenario scope", expanded=True):
        st.markdown(
            """
            <div class="section-note">
                Use these shared settings for both tabs.
                Labels are written in plain language to make each control easier to understand.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            base_fee = st.number_input(
                "Base trip fee",
                min_value=0.0,
                value=5.0,
                step=0.5,
                help="The fixed fee charged before environmental surcharge is added.",
            )
        with c2:
            emission_cost = st.number_input(
                "Environmental surcharge per kilogram of CO2",
                min_value=0.0,
                value=2.0,
                step=0.1,
                help="Additional cost for each kg of predicted CO2.",
            )
        with c3:
            target_emissions = st.number_input(
                "Target CO2 per trip (kg)",
                min_value=0.0,
                value=1.5,
                step=0.1,
                help="Used to show whether a trip is below or above your emissions target.",
            )
        with c4:
            target_fee = st.number_input(
                "Target trip fee",
                min_value=0.0,
                value=8.0,
                step=0.1,
                help="Used to show whether a trip is within your fee goal.",
            )

        c5, c6 = st.columns(2)
        with c5:
            selected_vehicles = st.multiselect(
                "Vehicle types to include in batch simulation",
                options=vehicle_choices,
                default=vehicle_choices,
            )
        with c6:
            selected_fuels = st.multiselect(
                "Fuel types to include in batch simulation",
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


def _display_columns_config():
    return {
        "trip_id": st.column_config.NumberColumn("Trip ID", format="%d"),
        "distance_km": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
        "vehicle_type": st.column_config.TextColumn("Vehicle"),
        "fuel_type": st.column_config.TextColumn("Fuel"),
        "predicted_emissions_kg": st.column_config.NumberColumn("Predicted CO2 (kg)", format="%.4f"),
        "calculated_fee": st.column_config.NumberColumn("Estimated fee", format="%.2f"),
        "emissions_per_km": st.column_config.NumberColumn("CO2 intensity (kg/km)", format="%.4f"),
    }


def trip_simulator_tab(
    vehicle_choices: list[str],
    fuel_choices: list[str],
    controls: dict,
):
    st.subheader("Trip Simulator")
    st.caption("Simulate one trip and see easy-to-read sustainability and fee indicators.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        trip_id = st.number_input("Trip identifier", min_value=1, value=1, step=1)
    with c2:
        distance_km = st.slider("Trip distance (km)", min_value=1.0, max_value=100.0, value=12.0, step=0.5)
    with c3:
        vehicle = st.selectbox("Vehicle type", options=vehicle_choices)
    with c4:
        fuel = st.selectbox("Fuel type", options=fuel_choices, index=min(2, len(fuel_choices) - 1))

    run_single = st.button("Run trip simulation", type="primary", use_container_width=True)

    if run_single:
        try:
            result = simulate_trip(
                trip_id=int(trip_id),
                distance_km=float(distance_km),
                vehicle_type=vehicle,
                fuel_type=fuel,
                base_fee=controls["base_fee"],
                emission_cost_per_kg=controls["emission_cost"],
            )
            st.session_state["single_trip_result"] = result
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            return

    result = st.session_state.get("single_trip_result")
    if not result:
        st.info("Enter trip details and click 'Run trip simulation' to view outputs.")
        return

    emissions_delta = result["predicted_emissions_kg"] - controls["target_emissions"]
    fee_delta = result["calculated_fee"] - controls["target_fee"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Predicted CO2",
        f"{result['predicted_emissions_kg']:.4f} kg",
        delta=f"{emissions_delta:+.3f} vs target",
        delta_color="inverse",
    )
    m2.metric(
        "Estimated trip fee",
        f"{result['calculated_fee']:.2f}",
        delta=f"{fee_delta:+.2f} vs target",
        delta_color="inverse",
    )
    m3.metric("CO2 intensity", f"{result['emissions_per_km']:.4f} kg/km")
    m4.metric("Distance", f"{result['distance_km']:.2f} km")

    ev_comparison_msg = "No EV comparison available."
    if result["fuel_type"] != "EV" and "EV" in fuel_choices:
        try:
            ev_result = simulate_trip(
                trip_id=result["trip_id"],
                distance_km=result["distance_km"],
                vehicle_type=result["vehicle_type"],
                fuel_type="EV",
                base_fee=controls["base_fee"],
                emission_cost_per_kg=controls["emission_cost"],
            )
            savings = result["predicted_emissions_kg"] - ev_result["predicted_emissions_kg"]
            if savings > 0:
                ev_comparison_msg = f"Switching to EV could reduce emissions by approximately {savings:.2f} kg."
            else:
                ev_comparison_msg = "For this input, EV does not reduce emissions in the current model."
        except Exception:
            ev_comparison_msg = "EV comparison could not be computed for this trip."

    left, right = st.columns([1.6, 1])
    with left:
        st.markdown("#### Trip output table")
        st.dataframe(
            pd.DataFrame([result]),
            use_container_width=True,
            hide_index=True,
            column_config=_display_columns_config(),
        )
    with right:
        st.markdown("#### Quick interpretation")
        st.write(f"- **CO2 target check:** {'Below target' if emissions_delta <= 0 else 'Above target'}")
        st.write(f"- **Fee target check:** {'Within target' if fee_delta <= 0 else 'Above target'}")
        st.write(f"- **EV comparison:** {ev_comparison_msg}")


def scenario_explorer_tab(controls: dict):
    st.subheader("Scenario Explorer")
    st.caption("Generate many trips at once and compare outcomes with polished tables and charts.")

    if not controls["selected_vehicles"] or not controls["selected_fuels"]:
        st.warning("Choose at least one vehicle type and one fuel type in policy settings.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        min_distance = st.number_input("Minimum distance (km)", min_value=1.0, value=5.0, step=1.0)
    with c2:
        max_distance = st.number_input("Maximum distance (km)", min_value=1.0, value=40.0, step=1.0)
    with c3:
        steps = st.number_input("Number of distance points", min_value=2, value=8, step=1)

    if min_distance > max_distance:
        st.error("Minimum distance must be less than or equal to maximum distance.")
        return

    distances = [float(x) for x in np.linspace(min_distance, max_distance, int(steps)).round(2).tolist()]

    run_batch = st.button("Generate scenario set", use_container_width=True)
    if run_batch:
        try:
            df = build_scenario_dataframe(
                distances=distances,
                vehicle_types=controls["selected_vehicles"],
                fuel_types=controls["selected_fuels"],
                base_fee=controls["base_fee"],
                emission_cost_per_kg=controls["emission_cost"],
            )
            st.session_state["scenario_df"] = df
        except Exception as exc:
            st.error(f"Batch simulation failed: {exc}")
            return

    df = st.session_state.get("scenario_df")
    if df is None or df.empty:
        st.info("Click 'Generate scenario set' to display scenario insights.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Scenarios generated", f"{len(df)}")
    k2.metric("Average CO2", f"{df['predicted_emissions_kg'].mean():.3f} kg")
    k3.metric("Average fee", f"{df['calculated_fee'].mean():.2f}")
    k4.metric("Lowest-CO2 fuel (average)", df.groupby("fuel_type")["predicted_emissions_kg"].mean().idxmin())

    st.markdown("#### Scenario results table")
    results_sorted = df.sort_values(["predicted_emissions_kg", "calculated_fee"]).reset_index(drop=True)
    st.dataframe(
        results_sorted,
        use_container_width=True,
        hide_index=True,
        column_config=_display_columns_config(),
    )

    st.download_button(
        "Download scenario results as CSV",
        data=results_sorted.to_csv(index=False).encode("utf-8"),
        file_name="scenario_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    chart1, chart2 = st.columns(2)
    with chart1:
        st.markdown("#### Average CO2 by fuel type")
        fuel_emissions = (
            df.groupby("fuel_type", as_index=False)["predicted_emissions_kg"]
            .mean()
            .sort_values("predicted_emissions_kg")
            .set_index("fuel_type")
        )
        st.bar_chart(fuel_emissions)

    with chart2:
        st.markdown("#### Average fee by vehicle type")
        vehicle_fee = (
            df.groupby("vehicle_type", as_index=False)["calculated_fee"]
            .mean()
            .sort_values("calculated_fee")
            .set_index("vehicle_type")
        )
        st.bar_chart(vehicle_fee)

    st.markdown("#### CO2 trend as distance increases")
    trend_df = (
        df.groupby(["distance_km", "fuel_type"], as_index=False)["predicted_emissions_kg"]
        .mean()
        .pivot(index="distance_km", columns="fuel_type", values="predicted_emissions_kg")
        .sort_index()
    )
    st.line_chart(trend_df)

    st.markdown("#### Top 10 lowest-emission scenarios")
    st.dataframe(
        df.nsmallest(10, "predicted_emissions_kg").reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config=_display_columns_config(),
    )


def guide_tab():
    st.subheader("How to use this dashboard")
    st.markdown(
        """
        1. Open **Policy settings and scenario scope** and set your fee policy and targets.  
        2. Use **Trip Simulator** for one specific trip.  
        3. Use **Scenario Explorer** to compare many combinations and export results.  
        4. Watch the target deltas to quickly see whether scenarios are above or below your goals.
        """
    )


def main():
    inject_theme()
    app_header()

    if model_load_error is not None:
        st.error(
            "Model is not ready. Install required dependencies and ensure model files exist.\n\n"
            f"Details: {model_load_error}"
        )
        st.stop()

    vehicle_choices, fuel_choices = _available_categories()
    controls = common_controls(vehicle_choices, fuel_choices)

    tab1, tab2, tab3 = st.tabs(["Trip Simulator", "Scenario Explorer", "Guide"])
    with tab1:
        trip_simulator_tab(vehicle_choices, fuel_choices, controls)
    with tab2:
        scenario_explorer_tab(controls)
    with tab3:
        guide_tab()


if __name__ == "__main__":
    main()
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
