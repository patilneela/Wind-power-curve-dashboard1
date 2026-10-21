import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Power Curve Dashboard", layout="wide")

WTG_COL = "Name"
TIME_COL = "Time"
STATUS_COL = "PCurveStsAve"
WIND_SPEED_COL = "WindSpeedAve"
WIND_STD_COL = "WindSpeedStd"
PITCH_STD_COL = "BladePitchStd"
POWER_COL = "ActivePWAve"
BIN_WIDTH = 0.5

REQUIRED_SCADA_COLUMNS = [
    WTG_COL,
    TIME_COL,
    STATUS_COL,
    WIND_SPEED_COL,
    WIND_STD_COL,
    PITCH_STD_COL,
    POWER_COL,
]


def ensure_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


@st.cache_data(show_spinner=False)
def load_scada(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    if not file_name.lower().endswith(".csv"):
        raise ValueError("SCADA input must be a CSV file.")

    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    ensure_required_columns(df, REQUIRED_SCADA_COLUMNS, "SCADA file")

    for col in [STATUS_COL, WIND_STD_COL, PITCH_STD_COL, WIND_SPEED_COL, POWER_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", dayfirst=True)
    df[WTG_COL] = df[WTG_COL].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_reference_curves(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    if not file_name.lower().endswith(".xlsx"):
        raise ValueError("Reference input must be an XLSX file.")

    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Power Curve", header=None)
    except ValueError:
        raise ValueError("The Excel file must contain a sheet named 'Power Curve'.")

    if raw.shape[0] < 4:
        raise ValueError("Reference workbook Power Curve sheet does not contain enough rows.")

    site_row = raw.iloc[2]
    wind_speed = pd.to_numeric(raw.iloc[3:, 0], errors="coerce")
    curves = {}

    for idx in range(1, raw.shape[1]):
        curve_name = site_row.iloc[idx]
        if pd.isna(curve_name):
            continue

        curve_name = str(curve_name).replace("\n", " ").strip()
        if not curve_name:
            continue

        power = pd.to_numeric(raw.iloc[3:, idx], errors="coerce")
        curve_df = pd.DataFrame(
            {
                "ReferenceWindSpeed_mps": wind_speed,
                "ReferencePower_kW": power,
            }
        ).dropna()

        curve_df = (
            curve_df.sort_values("ReferenceWindSpeed_mps")
            .drop_duplicates("ReferenceWindSpeed_mps")
            .reset_index(drop=True)
        )

        if not curve_df.empty:
            curves[curve_name] = curve_df

    if not curves:
        raise ValueError("No usable reference curves were found in the workbook.")

    return curves


def run_analysis(scada: pd.DataFrame, ref: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    working = scada.copy()

    mask_status = working[STATUS_COL].eq(100)
    mask_wind_std = working[WIND_STD_COL].lt(1)
    mask_pitch_std = working[PITCH_STD_COL].between(-5, 19, inclusive="both")
    mask = mask_status & mask_wind_std & mask_pitch_std
    filtered = working.loc[mask].copy()

    filter_counts = pd.DataFrame(
        [
            ["Total SCADA records", len(working)],
            ["PCurveStsAve = 100", int(mask_status.sum())],
            ["After PCurveStsAve = 100 and WindSpeedStd < 1", int((mask_status & mask_wind_std).sum())],
            ["After all three required filters", int(mask.sum())],
        ],
        columns=["Filter Stage", "Record Count"],
    )

    filtered = filtered.rename(
        columns={
            WTG_COL: "WTG",
            TIME_COL: "DateTime",
            WIND_SPEED_COL: "WindSpeed_mps",
            POWER_COL: "ActualPower_kW",
        }
    )

    ref_working = ref.copy()
    for col in ["ReferenceWindSpeed_mps", "ReferencePower_kW"]:
        ref_working[col] = pd.to_numeric(ref_working[col], errors="coerce")

    ref_working = (
        ref_working.dropna()
        .sort_values("ReferenceWindSpeed_mps")
        .drop_duplicates("ReferenceWindSpeed_mps")
        .reset_index(drop=True)
    )

    if ref_working.empty:
        raise ValueError("Selected reference curve has no valid numeric data.")

    ref_min = ref_working["ReferenceWindSpeed_mps"].min()
    ref_max = ref_working["ReferenceWindSpeed_mps"].max()

    filtered = filtered[filtered["WindSpeed_mps"].between(ref_min, ref_max)].copy()
    if filtered.empty:
        raise ValueError("No SCADA records remain after applying filters and reference wind-speed range.")

    filtered["ReferencePower_kW"] = np.interp(
        filtered["WindSpeed_mps"],
        ref_working["ReferenceWindSpeed_mps"],
        ref_working["ReferencePower_kW"],
    )
    filtered["Deviation_kW"] = filtered["ActualPower_kW"] - filtered["ReferencePower_kW"]
    filtered["Shortfall_kW"] = -filtered["Deviation_kW"]
    filtered["Deviation_pct"] = np.where(
        filtered["ReferencePower_kW"] > 0,
        100 * filtered["Deviation_kW"] / filtered["ReferencePower_kW"],
        np.nan,
    )
    filtered["WindSpeedBin_mps"] = np.floor(filtered["WindSpeed_mps"] / BIN_WIDTH) * BIN_WIDTH

    wtg_bins = (
        filtered.groupby(["WTG", "WindSpeedBin_mps"])
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanShortfall_kW=("Shortfall_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
            PowerStdDev_kW=("ActualPower_kW", "std"),
        )
        .reset_index()
    )

    wtg_summary = (
        filtered.groupby("WTG")
        .agg(
            ValidSamples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanSampleDeviation_pct=("Deviation_pct", "mean"),
            FirstTimestamp=("DateTime", "min"),
            LastTimestamp=("DateTime", "max"),
        )
        .reset_index()
    )

    wtg_summary["AggregateDeviation_pct"] = np.where(
        wtg_summary["MeanReferencePower_kW"] > 0,
        100 * (wtg_summary["MeanActualPower_kW"] - wtg_summary["MeanReferencePower_kW"])
        / wtg_summary["MeanReferencePower_kW"],
        np.nan,
    )
    wtg_summary = wtg_summary.sort_values("AggregateDeviation_pct")

    return {
        "filtered": filtered,
        "filter_counts": filter_counts,
        "wtg_bins": wtg_bins,
        "wtg_summary": wtg_summary,
        "ref": ref_working,
        "rows_passing": int(mask.sum()),
        "rows_compared": len(filtered),
        "wtg_count": int(filtered["WTG"].nunique()),
    }


def make_wtg_power_curve_plotly(
    ref: pd.DataFrame,
    filtered_wtg: pd.DataFrame,
    wtg_bins: pd.DataFrame,
    wtg_name: str,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=filtered_wtg["WindSpeed_mps"],
            y=filtered_wtg["ActualPower_kW"],
            mode="markers",
            name=f"{wtg_name} SCADA",
            marker=dict(size=5, opacity=0.18, color="#1f77b4"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ref["ReferenceWindSpeed_mps"],
            y=ref["ReferencePower_kW"],
            mode="lines",
            name="Reference Curve",
            line=dict(width=3, color="#d62728"),
        )
    )

    if not wtg_bins.empty:
        fig.add_trace(
            go.Scatter(
                x=wtg_bins["MeanWindSpeed_mps"],
                y=wtg_bins["MeanActualPower_kW"],
                mode="lines+markers",
                name=f"{wtg_name} mean by bin",
                line=dict(width=3, color="#2ca02c"),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=f"{wtg_name} Power Curve",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


st.title("Power Curve Dashboard")
st.write("Upload SCADA CSV and reference Excel file to see all turbine graphs after filtering.")

st.sidebar.header("Upload Files")
uploaded_scada = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])
uploaded_ref = st.sidebar.file_uploader("Upload Reference Excel", type=["xlsx"])

if uploaded_scada is None:
    st.info("Please upload the SCADA CSV file.")
    st.stop()

if uploaded_ref is None:
    st.info("Please upload the reference Excel file.")
    st.stop()

try:
    scada = load_scada(uploaded_scada.getvalue(), uploaded_scada.name)
    reference_curves = load_reference_curves(uploaded_ref.getvalue(), uploaded_ref.name)
except Exception as exc:
    st.exception(exc)
    st.stop()

curve_names = sorted(reference_curves.keys())
selected_curve = st.sidebar.selectbox("Select reference curve", curve_names)

scada_filtered = scada.copy()
valid_time = scada[TIME_COL].dropna()

if not valid_time.empty:
    min_date = valid_time.min().date()
    max_date = valid_time.max().date()

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        scada_filtered = scada_filtered[
            scada_filtered[TIME_COL].dt.date.between(start_date, end_date)
        ].copy()

if scada_filtered.empty:
    st.warning("No SCADA data available after date filtering.")
    st.stop()

try:
    results = run_analysis(scada_filtered, reference_curves[selected_curve])
except Exception as exc:
    st.exception(exc)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows passing filters", f"{results['rows_passing']:,}")
c2.metric("Rows compared", f"{results['rows_compared']:,}")
c3.metric("WTGs analysed", f"{results['wtg_count']:,}")

st.subheader("Filter Summary")
st.dataframe(results["filter_counts"], use_container_width=True, hide_index=True)

st.subheader("WTG Summary")
st.dataframe(results["wtg_summary"], use_container_width=True, hide_index=True)

all_wtgs = sorted(results["filtered"]["WTG"].dropna().unique().tolist())

if not all_wtgs:
    st.warning("No turbine data available after filtering.")
    st.stop()

st.subheader(f"All Turbine Graphs ({len(all_wtgs)})")

for wtg in all_wtgs:
    filtered_wtg = results["filtered"][results["filtered"]["WTG"] == wtg].copy()
    wtg_bins = results["wtg_bins"][results["wtg_bins"]["WTG"] == wtg].copy()

    if filtered_wtg.empty:
        continue

    st.markdown(f"### {wtg}")
    fig = make_wtg_power_curve_plotly(results["ref"], filtered_wtg, wtg_bins, wtg)
    st.plotly_chart(fig, use_container_width=True)
