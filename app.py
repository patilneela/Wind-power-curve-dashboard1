from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Power Curve Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
CSV_PATH = UPLOADS_DIR / "EN-156-3.3_20260801_20260826_ten_minutes-1792412779244.csv"
XLSX_PATH = UPLOADS_DIR / "All Sites Specific Power Curve data_ENV1-1792412779249.xlsx"

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


@st.cache_data(show_spinner=False)
def load_scada(csv_path: str) -> pd.DataFrame:
    scada = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_SCADA_COLUMNS if col not in scada.columns]
    if missing:
        raise ValueError(f"SCADA file is missing required columns: {', '.join(missing)}")

    for col in [STATUS_COL, WIND_STD_COL, PITCH_STD_COL, WIND_SPEED_COL, POWER_COL]:
        scada[col] = pd.to_numeric(scada[col], errors="coerce")

    scada[TIME_COL] = pd.to_datetime(scada[TIME_COL], errors="coerce", dayfirst=True)
    return scada


@st.cache_data(show_spinner=False)
def load_reference_curves(xlsx_path: str) -> dict[str, pd.DataFrame]:
    raw = pd.read_excel(xlsx_path, sheet_name="Power Curve", header=None)
    if raw.shape[0] < 4:
        raise ValueError("Reference workbook Power Curve sheet does not contain enough rows.")

    site_row = raw.iloc[2]
    curves: dict[str, pd.DataFrame] = {}

    wind_speed = pd.to_numeric(raw.iloc[3:, 0], errors="coerce")
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
        curve_df = curve_df.sort_values("ReferenceWindSpeed_mps").drop_duplicates(
            "ReferenceWindSpeed_mps"
        )

        if not curve_df.empty:
            curves[curve_name] = curve_df.reset_index(drop=True)

    if not curves:
        raise ValueError("No usable reference curves were found in the workbook.")
    return curves


def run_analysis(scada: pd.DataFrame, ref: pd.DataFrame) -> dict[str, pd.DataFrame | float | int]:
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
            [
                "After PCurveStsAve = 100 and WindSpeedStd < 1",
                int((mask_status & mask_wind_std).sum()),
            ],
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

    ref = ref.copy()
    for col in ["ReferenceWindSpeed_mps", "ReferencePower_kW"]:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    ref = ref.dropna().sort_values("ReferenceWindSpeed_mps").drop_duplicates(
        "ReferenceWindSpeed_mps"
    )
    if ref.empty:
        raise ValueError("Selected reference curve has no valid numeric data.")

    ref_min = ref["ReferenceWindSpeed_mps"].min()
    ref_max = ref["ReferenceWindSpeed_mps"].max()
    filtered = filtered[filtered["WindSpeed_mps"].between(ref_min, ref_max)].copy()

    if filtered.empty:
        raise ValueError("No SCADA records remain after applying filters and reference wind-speed range.")

    filtered["ReferencePower_kW"] = np.interp(
        filtered["WindSpeed_mps"],
        ref["ReferenceWindSpeed_mps"],
        ref["ReferencePower_kW"],
    )
    filtered["Deviation_kW"] = filtered["ActualPower_kW"] - filtered["ReferencePower_kW"]
    filtered["Shortfall_kW"] = -filtered["Deviation_kW"]
    filtered["Deviation_pct"] = np.where(
        filtered["ReferencePower_kW"] > 0,
        100 * filtered["Deviation_kW"] / filtered["ReferencePower_kW"],
        np.nan,
    )
    filtered["WindSpeedBin_mps"] = (
        np.floor(filtered["WindSpeed_mps"] / BIN_WIDTH) * BIN_WIDTH
    )

    fleet_bins = (
        filtered.groupby("WindSpeedBin_mps")
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            ["Total SCADA records", len(working)],
            ["Records passing all required filters", int(mask.sum())],
            ["Records compared to reference curve", len(filtered)],
            ["WTGs compared", filtered["WTG"].nunique()],
            ["Reference curve min wind speed (m/s)", ref_min],
            ["Reference curve max wind speed (m/s)", ref_max],
            ["Filter 1", "PCurveStsAve = 100"],
            ["Filter 2", "WindSpeedStd < 1"],
            ["Filter 3", "-5 <= BladePitchStd <= 19"],
            [
                "Deviation sign",
                "Actual power - reference power; negative means below reference",
            ],
        ],
        columns=["Metric", "Value"],
    )

    return {
        "filtered": filtered,
        "filter_counts": filter_counts,
        "fleet_bins": fleet_bins,
        "overall": overall,
        "ref": ref,
        "rows_passing": int(mask.sum()),
        "rows_compared": len(filtered),
        "wtg_count": int(filtered["WTG"].nunique()),
    }


def make_power_curve_figure(ref: pd.DataFrame, filtered: pd.DataFrame, fleet_bins: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        ref["ReferenceWindSpeed_mps"],
        ref["ReferencePower_kW"],
        linewidth=2.2,
        label="Reference Power Curve",
    )
    ax.scatter(
        filtered["WindSpeed_mps"],
        filtered["ActualPower_kW"],
        s=5,
        alpha=0.12,
        label="Filtered 10-min SCADA",
    )
    ax.plot(
        fleet_bins["MeanWindSpeed_mps"],
        fleet_bins["MeanActualPower_kW"],
        marker="o",
        linewidth=2,
        label="SCADA mean by 0.5 m/s bin",
    )
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Active Power (kW)")
    ax.set_title("Power Curve Analysis")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig



def make_deviation_figure(fleet_bins: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(0, linewidth=1, color="black")
    ax.plot(
        fleet_bins["MeanWindSpeed_mps"],
        fleet_bins["MeanDeviation_kW"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Mean Deviation (Actual - Reference, kW)")
    ax.set_title("Power Deviation by Wind-Speed Bin")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


st.title("Power Curve Dashboard")
st.caption("10-minute static file vs selectable site-specific reference power curve")

missing_files = [str(path.name) for path in [CSV_PATH, XLSX_PATH] if not path.exists()]
if missing_files:
    st.error(f"Missing required input files: {', '.join(missing_files)}")
    st.stop()

try:
    scada = load_scada(str(CSV_PATH))
    reference_curves = load_reference_curves(str(XLSX_PATH))
except Exception as exc:
    st.error(f"Failed to load input data: {exc}")
    st.stop()

curve_names = sorted(reference_curves.keys())
selected_curve = st.selectbox("Select reference power curve", curve_names)

ref_df = reference_curves[selected_curve]

try:
    results = run_analysis(scada, ref_df)
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.stop()

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Rows passing filters", f"{results['rows_passing']:,}")
metric_col2.metric("Rows compared to curve", f"{results['rows_compared']:,}")
metric_col3.metric("WTGs analysed", f"{results['wtg_count']:,}")

st.subheader("Filter counts")
st.dataframe(results["filter_counts"], width="stretch", hide_index=True)

st.subheader("Overall summary")
overall_display = results["overall"].astype({"Value": "string"})
st.dataframe(overall_display, width="stretch", hide_index=True)

st.subheader("Power curve comparison")
power_curve_fig = make_power_curve_figure(
    results["ref"], results["filtered"], results["fleet_bins"]
)
st.pyplot(power_curve_fig, width="stretch")
plt.close(power_curve_fig)

st.subheader("Deviation by wind-speed bin")
deviation_fig = make_deviation_figure(results["fleet_bins"])
st.pyplot(deviation_fig, width="stretch")
plt.close(deviation_fig)

with st.expander("Show fleet wind bin summary"):
    st.dataframe(results["fleet_bins"], width="stretch", hide_index=True)

with st.expander("Show filtered SCADA sample"):
    preview_cols = [
        "WTG",
        "DateTime",
        STATUS_COL,
        WIND_STD_COL,
        PITCH_STD_COL,
        "WindSpeed_mps",
        "ActualPower_kW",
        "ReferencePower_kW",
        "Deviation_kW",
        "Deviation_pct",
        "WindSpeedBin_mps",
    ]
    st.dataframe(
        results["filtered"][preview_cols].head(200),
        width="stretch",
        hide_index=True,
    )
