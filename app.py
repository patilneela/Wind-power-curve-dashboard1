import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
import io
import os
import re
import math

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Power Curve Analytics Report",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================

BIN_SIZE = 0.5
DEFAULT_CAPACITY_KW = 3300.0

MIN_VALID_ROWS = 30

WIND_MIN = 3.0
WIND_MAX = 25.0

PITCH_MIN = -5.0
PITCH_MAX = 5.0

NORMAL_LOW = -2.0
NORMAL_HIGH = 2.0

# Number of graphs per page
GRAPHS_PER_PAGE = 6

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    h1 {
        font-size: 2.1rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    .metric-card {
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 14px;
        background: white;
        text-align: center;
        min-height: 100px;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #666;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }

    .status-normal {
        background: #d9f2d9;
        color: #176b17;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
    }

    .status-warning {
        background: #fff0cc;
        color: #8a5a00;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
    }

    .status-danger {
        background: #ffdede;
        color: #9b0000;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
    }

    .status-na {
        background: #eeeeee;
        color: #555;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
    }

    .small-note {
        color: #666;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "scada_df" not in st.session_state:
    st.session_state.scada_df = None

if "reference_bytes" not in st.session_state:
    st.session_state.reference_bytes = None

if "reference_name" not in st.session_state:
    st.session_state.reference_name = None

if "site_master_df" not in st.session_state:
    st.session_state.site_master_df = None

if "results" not in st.session_state:
    st.session_state.results = {}

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def turbine_sort_number(name):
    """
    Safely extracts a turbine number.

    Works with:
        LOC-000
        LOC-001
        LOC001
        WTG01
        WTG-01
        Turbine 31
    """

    s = clean_name(name)

    numbers = re.findall(r"\d+", s)

    if numbers:
        try:
            return int(numbers[-1])
        except Exception:
            pass

    return 999999


def normalize_column_name(col):
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(df, patterns):
    """
    Find a column based on keywords.
    """

    cols = list(df.columns)

    # Exact normalized matching first
    normalized = {
        normalize_column_name(c): c
        for c in cols
    }

    for pattern in patterns:
        p = normalize_column_name(pattern)

        if p in normalized:
            return normalized[p]

    # Partial matching
    for c in cols:
        nc = normalize_column_name(c)

        for pattern in patterns:
            p = normalize_column_name(pattern)

            if p in nc:
                return c

    return None


def detect_turbine_column(df):
    candidates = [
        "Name",
        "Turbine",
        "TurbineName",
        "Turbine Name",
        "WTG",
        "WTGName",
        "WindTurbine",
        "DeviceName"
    ]

    col = find_column(df, candidates)

    if col:
        return col

    # Fallback: find column containing name/turbine
    for c in df.columns:
        text = normalize_column_name(c)

        if "turbine" in text or text == "name" or "wtg" in text:
            return c

    return None


def detect_wind_column(df):
    candidates = [
        "WindSpeedAve",
        "WindSpeed",
        "Wind Speed",
        "WindSpeedAverage",
        "WindSpeedAvg",
        "Wind"
    ]

    return find_column(df, candidates)


def detect_power_column(df):
    candidates = [
        "ActivePWAve",
        "ActivePower",
        "Active Power",
        "ActivePowerAve",
        "ActivePowerAverage",
        "Power",
        "PowerAve",
        "GeneratedPower"
    ]

    return find_column(df, candidates)


def detect_pitch_column(df):
    candidates = [
        "BldPitch1Ave",
        "BladePitch1",
        "BladePitch1Ave",
        "Pitch",
        "PitchAve",
        "BldPitch"
    ]

    return find_column(df, candidates)


def detect_time_column(df):
    candidates = [
        "Time",
        "Timestamp",
        "DateTime",
        "Date Time",
        "TimeStamp",
        "Date",
        "UTC",
        "LocalTime"
    ]

    return find_column(df, candidates)


# ============================================================
# LOAD SCADA
# ============================================================

@st.cache_data(show_spinner=False)
def load_scada_bytes(file_bytes):

    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            low_memory=False
        )
    except Exception:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                engine="python"
            )
        except Exception as e:
            raise ValueError(
                f"Unable to read SCADA CSV: {e}"
            )

    df.columns = [str(c).strip() for c in df.columns]

    turbine_col = detect_turbine_column(df)
    wind_col = detect_wind_column(df)
    power_col = detect_power_column(df)
    pitch_col = detect_pitch_column(df)
    time_col = detect_time_column(df)

    if turbine_col is None:
        raise ValueError(
            "Could not detect turbine column. "
            "Expected something like Name, Turbine, WTG or TurbineName."
        )

    if wind_col is None:
        raise ValueError(
            "Could not detect wind-speed column."
        )

    if power_col is None:
        raise ValueError(
            "Could not detect active-power column."
        )

    # Rename essential columns
    rename_map = {
        turbine_col: "_Turbine",
        wind_col: "_Wind",
        power_col: "_Power"
    }

    if pitch_col:
        rename_map[pitch_col] = "_Pitch"

    if time_col:
        rename_map[time_col] = "_Time"

    df = df.rename(columns=rename_map)

    df["_Turbine"] = df["_Turbine"].astype(str).str.strip()

    df["_Wind"] = pd.to_numeric(
        df["_Wind"],
        errors="coerce"
    )

    df["_Power"] = pd.to_numeric(
        df["_Power"],
        errors="coerce"
    )

    if "_Pitch" in df.columns:
        df["_Pitch"] = pd.to_numeric(
            df["_Pitch"],
            errors="coerce"
        )

    if "_Time" in df.columns:

        df["_Time"] = pd.to_datetime(
            df["_Time"],
            errors="coerce"
        )

    return df


# ============================================================
# REFERENCE EXCEL READER
# ============================================================

def get_excel_sheets(file_bytes):
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        return xls.sheet_names
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def read_reference_excel(file_bytes):

    xls = pd.ExcelFile(
        io.BytesIO(file_bytes)
    )

    sheets = xls.sheet_names

    data = {}

    for sheet in sheets:
        try:
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                sheet_name=sheet,
                header=None
            )

            data[sheet] = df

        except Exception:
            continue

    return data


def detect_sites_from_reference(reference_data):

    sites = []

    for sheet_name, df in reference_data.items():

        if df.empty:
            continue

        # Search every cell for likely site names.
        for col in df.columns:

            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
            )

            for value in values:

                value_clean = value.strip()

                if len(value_clean) < 3:
                    continue

                lower = value_clean.lower()

                # Ignore generic headers
                ignored = [
                    "wind speed",
                    "wind speed (m/s)",
                    "power",
                    "power (kw)",
                    "power curve",
                    "standard",
                    "theoretical",
                    "site",
                    "name",
                    "turbine"
                ]

                if lower in ignored:
                    continue

                # A site name is generally textual
                # and contains letters.
                if any(ch.isalpha() for ch in value_clean):

                    # Avoid extremely long sentences
                    if len(value_clean) <= 100:

                        sites.append(value_clean)

    # Remove duplicates safely
    unique_sites = []

    seen = set()

    for site in sites:

        key = site.lower().strip()

        if key not in seen:
            seen.add(key)
            unique_sites.append(site)

    return unique_sites


def site_matches_text(text, site):

    if text is None:
        return False

    text = str(text).strip().lower()
    site = str(site).strip().lower()

    if not text or not site:
        return False

    return site in text or text in site


# ============================================================
# REFERENCE CURVE EXTRACTION
# ============================================================

def numeric_series(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def extract_reference_curve(
    reference_data,
    selected_site
):

    best_curve = None
    best_score = -1

    for sheet_name, raw_df in reference_data.items():

        if raw_df.empty:
            continue

        df = raw_df.copy()

        # ----------------------------------------------------
        # METHOD 1:
        # Search rows containing site name
        # ----------------------------------------------------

        for r in range(len(df)):

            row_values = df.iloc[r].astype(str).tolist()

            row_text = " | ".join(row_values)

            if not site_matches_text(
                row_text,
                selected_site
            ):
                continue

            # Check every possible pair of adjacent columns
            for c in range(len(df.columns) - 1):

                wind_values = numeric_series(
                    df.iloc[r + 1:, c]
                )

                power_values = numeric_series(
                    df.iloc[r + 1:, c + 1]
                )

                mask = (
                    wind_values.notna()
                    & power_values.notna()
                )

                w = wind_values[mask]
                p = power_values[mask]

                if len(w) < 5:
                    continue

                # Physical wind range
                valid = (
                    (w >= 0)
                    & (w <= 30)
                    & (p >= 0)
                )

                w = w[valid]
                p = p[valid]

                if len(w) < 5:
                    continue

                score = len(w)

                if score > best_score:

                    curve = pd.DataFrame({
                        "Wind": w.values,
                        "ReferencePower": p.values
                    })

                    best_curve = curve
                    best_score = score

        # ----------------------------------------------------
        # METHOD 2:
        # Look for site name anywhere and nearby numeric columns
        # ----------------------------------------------------

        if best_curve is not None:
            continue

    if best_curve is None:

        # ----------------------------------------------------
        # FALLBACK:
        # Search columns with "wind" and "power"
        # ----------------------------------------------------

        for sheet_name, raw_df in reference_data.items():

            df = raw_df.copy()

            for c in range(len(df.columns)):

                col_text = str(
                    df.columns[c]
                ).lower()

                if "wind" not in col_text:
                    continue

                for pcol in range(len(df.columns)):

                    ptext = str(
                        df.columns[pcol]
                    ).lower()

                    if "power" not in ptext:
                        continue

                    w = numeric_series(
                        df.iloc[:, c]
                    )

                    p = numeric_series(
                        df.iloc[:, pcol]
                    )

                    mask = (
                        w.notna()
                        & p.notna()
                        & (w >= 0)
                        & (w <= 30)
                        & (p >= 0)
                    )

                    if mask.sum() >= 5:

                        best_curve = pd.DataFrame({
                            "Wind": w[mask].values,
                            "ReferencePower": p[mask].values
                        })

                        best_score = mask.sum()

                        break

                if best_curve is not None:
                    break

            if best_curve is not None:
                break

    if best_curve is None:
        return None

    # --------------------------------------------------------
    # Clean reference
    # --------------------------------------------------------

    best_curve["Wind"] = pd.to_numeric(
        best_curve["Wind"],
        errors="coerce"
    )

    best_curve["ReferencePower"] = pd.to_numeric(
        best_curve["ReferencePower"],
        errors="coerce"
    )

    best_curve = best_curve.dropna()

    best_curve = best_curve[
        (best_curve["Wind"] >= 0)
        & (best_curve["Wind"] <= 30)
        & (best_curve["ReferencePower"] >= 0)
    ]

    if best_curve.empty:
        return None

    # Aggregate duplicate wind points
    best_curve = (
        best_curve
        .groupby("Wind", as_index=False)
        ["ReferencePower"]
        .mean()
    )

    best_curve = best_curve.sort_values(
        "Wind"
    )

    return best_curve.reset_index(drop=True)


# ============================================================
# CREATE STANDARD REFERENCE CURVE
# ============================================================

def interpolate_reference(reference_curve):

    if reference_curve is None:
        return None

    if reference_curve.empty:
        return None

    min_wind = 4.0
    max_wind = 15.0

    bins = np.arange(
        min_wind,
        max_wind + BIN_SIZE / 2,
        BIN_SIZE
    )

    x = reference_curve["Wind"].values
    y = reference_curve["ReferencePower"].values

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # Remove duplicate x values
    unique_x, unique_idx = np.unique(
        x,
        return_index=True
    )

    unique_y = y[unique_idx]

    if len(unique_x) < 2:
        return None

    values = np.interp(
        bins,
        unique_x,
        unique_y
    )

    result = pd.DataFrame({
        "Wind": bins,
        "ReferencePower": values
    })

    return result


# ============================================================
# SCADA PROCESSING
# ============================================================

def process_turbine(
    turbine_df,
    reference_curve
):

    result = {
        "status": "No Data",
        "deviation": np.nan,
        "availability": np.nan,
        "raw_rows": 0,
        "valid_rows": 0,
        "std_power": np.nan,
        "actual_curve": None,
        "reference_curve": reference_curve,
        "comment": "No SCADA rows are available for this turbine in the selected date range."
    }

    if turbine_df is None or turbine_df.empty:
        return result

    result["raw_rows"] = len(turbine_df)

    df = turbine_df.copy()

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    df["_Wind"] = pd.to_numeric(
        df["_Wind"],
        errors="coerce"
    )

    df["_Power"] = pd.to_numeric(
        df["_Power"],
        errors="coerce"
    )

    mask = (
        df["_Wind"].notna()
        & df["_Power"].notna()
        & (df["_Wind"] >= WIND_MIN)
        & (df["_Wind"] <= WIND_MAX)
        & (df["_Power"] > 0)
    )

    # Pitch filter only when available
    if "_Pitch" in df.columns:

        pitch = pd.to_numeric(
            df["_Pitch"],
            errors="coerce"
        )

        pitch_mask = (
            pitch.isna()
            | (
                (pitch >= PITCH_MIN)
                & (pitch <= PITCH_MAX)
            )
        )

        mask = mask & pitch_mask

    valid = df[mask].copy()

    result["valid_rows"] = len(valid)

    if valid.empty:

        result["status"] = "No Curve"
        result["comment"] = (
            "SCADA rows are available, but no valid "
            "power-curve points remain after filtering."
        )

        return result

    if len(valid) < MIN_VALID_ROWS:

        result["status"] = "Insufficient Data"
        result["comment"] = (
            f"Only {len(valid)} valid SCADA rows are available. "
            f"At least {MIN_VALID_ROWS} rows are recommended."
        )

        # Still create curve if possible
        # so user can inspect data.

    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    result["std_power"] = float(
        valid["_Power"].std()
    ) if len(valid) > 1 else 0.0

    # --------------------------------------------------------
    # Wind binning
    # --------------------------------------------------------

    valid["_WindBin"] = (
        np.floor(
            valid["_Wind"] / BIN_SIZE
        ) * BIN_SIZE
        + BIN_SIZE / 2
    )

    # Round for clean labels
    valid["_WindBin"] = valid["_WindBin"].round(2)

    actual = (
        valid
        .groupby("_WindBin", as_index=False)
        ["_Power"]
        .mean()
        .rename(
            columns={
                "_WindBin": "Wind",
                "_Power": "ActualPower"
            }
        )
    )

    actual = actual.sort_values(
        "Wind"
    ).reset_index(drop=True)

    if len(actual) == 0:

        result["status"] = "No Curve"

        return result

    # --------------------------------------------------------
    # Smooth curve
    # --------------------------------------------------------

    if len(actual) >= 7:

        window = min(
            7,
            len(actual)
        )

        if window % 2 == 0:
            window -= 1

        if window >= 5:

            try:

                actual["ActualPowerSmooth"] = (
                    savgol_filter(
                        actual["ActualPower"].values,
                        window_length=window,
                        polyorder=2
                    )
                )

            except Exception:

                actual["ActualPowerSmooth"] = (
                    actual["ActualPower"]
                )

        else:

            actual["ActualPowerSmooth"] = (
                actual["ActualPower"]
            )

    else:

        actual["ActualPowerSmooth"] = (
            actual["ActualPower"]
        )

    # --------------------------------------------------------
    # Deviation calculation
    # --------------------------------------------------------

    if reference_curve is not None and not reference_curve.empty:

        ref_x = reference_curve["Wind"].values
        ref_y = reference_curve["ReferencePower"].values

        actual_x = actual["Wind"].values

        reference_at_actual = np.interp(
            actual_x,
            ref_x,
            ref_y
        )

        actual["ReferencePower"] = (
            reference_at_actual
        )

        # Only compare where reference > 0
        compare_mask = (
            actual["ReferencePower"] > 0
        )

        if compare_mask.sum() > 0:

            actual_mean = (
                actual.loc[
                    compare_mask,
                    "ActualPowerSmooth"
                ].mean()
            )

            reference_mean = (
                actual.loc[
                    compare_mask,
                    "ReferencePower"
                ].mean()
            )

            if reference_mean != 0:

                deviation = (
                    (
                        actual_mean
                        - reference_mean
                    )
                    / reference_mean
                    * 100
                )

                result["deviation"] = float(
                    deviation
                )

    result["actual_curve"] = actual

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    dev = result["deviation"]

    if pd.isna(dev):

        result["status"] = (
            "Insufficient Data"
            if len(valid) < MIN_VALID_ROWS
            else "No Curve"
        )

        result["comment"] = (
            "A valid power curve was created, "
            "but deviation could not be calculated "
            "against the reference curve."
        )

        return result

    # --------------------------------------------------------
    # Comments
    # --------------------------------------------------------

    if dev < -72:

        status = "Critical Underperformance"

        comment = (
            "Extreme underperformance. "
            "Check SCADA quality, turbine availability, "
            "curtailment, sensor signals and operating condition."
        )

    elif dev < -10:

        status = "High Underperformance"

        comment = (
            "Significant underperformance. Possible causes "
            "include blade condition, dust, yaw misalignment, "
            "control issues or availability losses."
        )

    elif dev < -2:

        status = "Underperformance"

        comment = (
            "Performance is below the reference curve. "
            "Check control, availability, blade and yaw conditions."
        )

    elif dev > 72:

        status = "Critical Overperformance"

        comment = (
            "Extreme overperformance. Check reference curve, "
            "SCADA scaling, sensor calibration and data quality."
        )

    elif dev > 8:

        status = "High Overperformance"

        comment = (
            "High overperformance. Possible causes include "
            "measurement/scaling issue, NTF mismatch or "
            "sensor alignment differences."
        )

    elif dev > 2:

        status = "Slight Overperformance"

        comment = (
            "Slight overperformance compared with the "
            "reference curve. Verify measurement and reference data."
        )

    else:

        status = "Normal Performance"

        comment = (
            "Performance is within the normal ±2% deviation range."
        )

    # If data is insufficient, retain that information
    if len(valid) < MIN_VALID_ROWS:

        status = "Insufficient Data"

        comment = (
            f"{len(valid)} valid SCADA rows were available. "
            "The calculated deviation should be treated cautiously."
        )

    result["status"] = status
    result["comment"] = comment

    return result


# ============================================================
# BUILD RESULT FOR ALL TURBINES
# ============================================================

def analyze_all_turbines(
    scada_df,
    reference_curve,
    turbines
):

    results = {}

    for turbine in turbines:

        turbine_df = scada_df[
            scada_df["_Turbine"].astype(str)
            == str(turbine)
        ].copy()

        results[turbine] = process_turbine(
            turbine_df,
            reference_curve
        )

    return results


# ============================================================
# STATUS PRIORITY
# ============================================================

def status_priority(status):

    status = str(status)

    priority = {
        "Normal Performance": 1,
        "Slight Overperformance": 2,
        "High Overperformance": 3,
        "Critical Overperformance": 4,
        "Underperformance": 5,
        "High Underperformance": 6,
        "Critical Underperformance": 7,
        "Insufficient Data": 8,
        "No Curve": 9,
        "No Data": 10
    }

    return priority.get(
        status,
        99
    )


# ============================================================
# CREATE RANKING DATAFRAME
# ============================================================

def create_ranking(results):

    rows = []

    for turbine, result in results.items():

        deviation = result.get(
            "deviation",
            np.nan
        )

        rows.append({
            "Turbine": str(turbine),
            "Deviation_%": deviation,
            "Status": result.get(
                "status",
                "No Data"
            ),
            "Valid Rows": result.get(
                "valid_rows",
                0
            ),
            "Raw Rows": result.get(
                "raw_rows",
                0
            ),
            "Std Dev (kW)": result.get(
                "std_power",
                np.nan
            ),
            "Comment": result.get(
                "comment",
                ""
            ),
            "_StatusSort": status_priority(
                result.get(
                    "status",
                    "No Data"
                )
            ),
            "_TurbineSort": turbine_sort_number(
                turbine
            )
        })

    ranking = pd.DataFrame(rows)

    if ranking.empty:
        return ranking

    # --------------------------------------------------------
    # VERY IMPORTANT:
    # Force numeric columns to numeric.
    # This prevents the pandas categorical sorting error.
    # --------------------------------------------------------

    ranking["Deviation_%"] = pd.to_numeric(
        ranking["Deviation_%"],
        errors="coerce"
    )

    ranking["_StatusSort"] = pd.to_numeric(
        ranking["_StatusSort"],
        errors="coerce"
    ).fillna(999)

    ranking["_TurbineSort"] = pd.to_numeric(
        ranking["_TurbineSort"],
        errors="coerce"
    ).fillna(999999)

    # --------------------------------------------------------
    # SAFE SORT
    # --------------------------------------------------------

    ranking = ranking.sort_values(
        by=[
            "_StatusSort",
            "Deviation_%",
            "_TurbineSort"
        ],
        ascending=[
            True,
            False,
            True
        ],
        na_position="last"
    ).reset_index(drop=True)

    # Rank only valid curves
    ranking["Rank"] = np.nan

    valid_mask = ranking["Deviation_%"].notna()

    ranking.loc[
        valid_mask,
        "Rank"
    ] = np.arange(
        1,
        valid_mask.sum() + 1
    )

    return ranking


# ============================================================
# GRAPH
# ============================================================

def create_power_curve_figure(
    turbine,
    result
):

    fig = go.Figure()

    actual = result.get(
        "actual_curve"
    )

    reference = result.get(
        "reference_curve"
    )

    # --------------------------------------------------------
    # Reference curve
    # --------------------------------------------------------

    if (
        reference is not None
        and not reference.empty
    ):

        fig.add_trace(
            go.Scatter(
                x=reference["Wind"],
                y=reference["ReferencePower"],
                mode="lines",
                name="Reference",
                line=dict(
                    color="red",
                    width=3,
                    dash="dash"
                ),
                hovertemplate=(
                    "Wind: %{x:.2f} m/s"
                    "<br>"
                    "Reference: %{y:.0f} kW"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Actual raw curve
    # --------------------------------------------------------

    if (
        actual is not None
        and not actual.empty
    ):

        fig.add_trace(
            go.Scatter(
                x=actual["Wind"],
                y=actual["ActualPower"],
                mode="markers",
                name="Actual Raw",
                marker=dict(
                    size=7,
                    opacity=0.35
                ),
                hovertemplate=(
                    "Wind: %{x:.2f} m/s"
                    "<br>"
                    "Actual: %{y:.0f} kW"
                    "<extra></extra>"
                )
            )
        )

        # ----------------------------------------------------
        # Smoothed actual curve
        # ----------------------------------------------------

        smooth_col = (
            "ActualPowerSmooth"
            if "ActualPowerSmooth" in actual.columns
            else "ActualPower"
        )

        fig.add_trace(
            go.Scatter(
                x=actual["Wind"],
                y=actual[smooth_col],
                mode="lines+markers",
                name="Actual Curve",
                line=dict(
                    color="green",
                    width=3
                ),
                marker=dict(
                    size=6
                ),
                hovertemplate=(
                    "Wind: %{x:.2f} m/s"
                    "<br>"
                    "Actual: %{y:.0f} kW"
                    "<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # Deviation
    # --------------------------------------------------------

    dev = result.get(
        "deviation",
        np.nan
    )

    status = result.get(
        "status",
        "No Data"
    )

    if pd.isna(dev):
        dev_text = "N/A"
    else:
        dev_text = f"{dev:.2f}%"

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = (
        f"<b>{turbine}</b>"
        f"<br>"
        f"<sup>{status} | Deviation: {dev_text}</sup>"
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            title="Wind Speed (m/s)",
            range=[2.5, 16],
            dtick=1,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Power (kW)",
            rangemode="tozero",
            showgrid=True,
            zeroline=False
        ),
        height=480,
        margin=dict(
            l=65,
            r=30,
            t=85,
            b=65
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        )
    )

    return fig


# ============================================================
# DATE FILTER
# ============================================================

def apply_date_filter(df):

    if "_Time" not in df.columns:

        return df, None, None

    valid_time = df["_Time"].dropna()

    if valid_time.empty:

        return df, None, None

    min_date = valid_time.min().date()
    max_date = valid_time.max().date()

    st.sidebar.markdown("### Date Range")

    date_option = st.sidebar.selectbox(
        "Date Option",
        [
            "Clear",
            "Today",
            "This Week",
            "This Month",
            "Last Week",
            "Last Month",
            "Manual"
        ],
        index=6
    )

    selected_start = None
    selected_end = None

    if date_option == "Clear":

        return df, min_date, max_date

    elif date_option == "Today":

        selected_start = max_date
        selected_end = max_date

    elif date_option == "This Week":

        selected_end = max_date
        selected_start = (
            selected_end
            - timedelta(
                days=selected_end.weekday()
            )
        )

    elif date_option == "This Month":

        selected_end = max_date
        selected_start = selected_end.replace(
            day=1
        )

    elif date_option == "Last Week":

        this_monday = (
            max_date
            - timedelta(
                days=max_date.weekday()
            )
        )

        selected_end = (
            this_monday
            - timedelta(days=1)
        )

        selected_start = (
            selected_end
            - timedelta(days=6)
        )

    elif date_option == "Last Month":

        first_this_month = max_date.replace(
            day=1
        )

        selected_end = (
            first_this_month
            - timedelta(days=1)
        )

        selected_start = selected_end.replace(
            day=1
        )

    elif date_option == "Manual":

        default_start = max(
            min_date,
            max_date - timedelta(days=15)
        )

        selected_range = st.sidebar.date_input(
            "Select date range",
            value=(
                default_start,
                max_date
            ),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(
            selected_range,
            tuple
        ) and len(selected_range) == 2:

            selected_start = selected_range[0]
            selected_end = selected_range[1]

        else:

            selected_start = selected_range
            selected_end = selected_range

    if selected_start is None:
        return df, min_date, max_date

    mask = (
        df["_Time"].dt.date >= selected_start
    ) & (
        df["_Time"].dt.date <= selected_end
    )

    filtered = df[mask].copy()

    return (
        filtered,
        selected_start,
        selected_end
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("⚡ Power Curve Analytics Report")

st.caption(
    "SCADA Power Curve Analysis | Reference vs Actual Performance"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Upload Input Files")

scada_file = st.sidebar.file_uploader(
    "1. Upload SCADA CSV",
    type=["csv"],
    key="scada_upload"
)

reference_file = st.sidebar.file_uploader(
    "2. Upload Reference Excel",
    type=["xlsx", "xls"],
    key="reference_upload"
)

# ============================================================
# LOAD SCADA
# ============================================================

if scada_file is not None:

    try:

        file_bytes = scada_file.getvalue()

        scada_df = load_scada_bytes(
            file_bytes
        )

        st.session_state.scada_df = scada_df

    except Exception as e:

        st.error(
            f"SCADA loading error: {e}"
        )

        st.stop()

elif st.session_state.scada_df is None:

    st.info(
        "Please upload a SCADA CSV file from the sidebar."
    )

    st.stop()

else:

    scada_df = st.session_state.scada_df


scada_df = st.session_state.scada_df

# ============================================================
# LOAD REFERENCE
# ============================================================

if reference_file is not None:

    try:

        reference_bytes = reference_file.getvalue()

        st.session_state.reference_bytes = (
            reference_bytes
        )

        st.session_state.reference_name = (
            reference_file.name
        )

    except Exception as e:

        st.error(
            f"Reference file error: {e}"
        )

        st.stop()


if st.session_state.reference_bytes is None:

    st.warning(
        "Please upload the Reference Excel file."
    )

    st.stop()


reference_data = read_reference_excel(
    st.session_state.reference_bytes
)

# ============================================================
# TURBINE DETECTION
# ============================================================

scada_df["_Turbine"] = (
    scada_df["_Turbine"]
    .astype(str)
    .str.strip()
)

all_turbines = [
    x for x in
    scada_df["_Turbine"].dropna().unique()
    if str(x).strip()
]

# SAFE TURBINE SORT
all_turbines = sorted(
    all_turbines,
    key=lambda x: (
        turbine_sort_number(x),
        str(x).lower()
    )
)

# ============================================================
# SITE DETECTION
# ============================================================

detected_sites = detect_sites_from_reference(
    reference_data
)

if not detected_sites:

    st.warning(
        "No site names could be automatically detected "
        "from the reference Excel."
    )

    # Provide sheet selection as fallback
    sheet_names = list(
        reference_data.keys()
    )

    if sheet_names:

        st.sidebar.markdown(
            "### Reference Sheet"
        )

        selected_sheet = st.sidebar.selectbox(
            "Select Reference Sheet",
            sheet_names
        )

        st.info(
            f"Using reference sheet: {selected_sheet}"
        )

        # Use first sheet as fallback site
        selected_site = selected_sheet

    else:

        st.error(
            "No readable sheets were found in the reference Excel."
        )

        st.stop()

else:

    # Remove obvious headers from detected sites
    cleaned_sites = []

    for site in detected_sites:

        low = site.lower().strip()

        if low in [
            "site",
            "site name",
            "wind speed",
            "wind speed (m/s)",
            "power",
            "power (kw)",
            "standard",
            "theoretical"
        ]:
            continue

        cleaned_sites.append(site)

    detected_sites = cleaned_sites

    # If duplicate site names differ only slightly,
    # keep unique values.
    detected_sites = list(
        dict.fromkeys(
            detected_sites
        )
    )

    st.sidebar.markdown(
        "### Select Site"
    )

    selected_site = st.sidebar.selectbox(
        "Site",
        detected_sites
    )

# ============================================================
# VIEW SELECTION
# ============================================================

st.sidebar.markdown(
    "### Select View"
)

view_mode = st.sidebar.radio(
    "",
    [
        "Single Turbine",
        "Compare Turbines",
        "Show All Turbines"
    ]
)

# ============================================================
# DATE FILTER
# ============================================================

filtered_scada, start_date, end_date = (
    apply_date_filter(
        scada_df
    )
)

if start_date is not None:

    st.sidebar.success(
        f"Applied Range: {start_date} → {end_date}"
    )

# ============================================================
# TURBINE COUNT
# ============================================================

filtered_turbines = [
    x for x in
    filtered_scada["_Turbine"]
    .dropna()
    .astype(str)
    .unique()
]

# Important:
# Detect turbines from the COMPLETE uploaded SCADA,
# not only the filtered rows.
#
# This prevents turbines disappearing simply because
# they have no rows in the selected date range.

all_uploaded_turbines = [
    x for x in
    scada_df["_Turbine"]
    .dropna()
    .astype(str)
    .unique()
]

all_uploaded_turbines = sorted(
    all_uploaded_turbines,
    key=lambda x: (
        turbine_sort_number(x),
        x.lower()
    )
)

st.sidebar.info(
    f"{len(all_uploaded_turbines)} turbines detected "
    f"in uploaded SCADA"
)

# ============================================================
# REFERENCE CURVE
# ============================================================

with st.spinner(
    "Loading reference power curve..."
):

    raw_reference = extract_reference_curve(
        reference_data,
        selected_site
    )

reference_curve = interpolate_reference(
    raw_reference
)

if reference_curve is None:

    st.warning(
        f"Reference curve could not be extracted for "
        f"'{selected_site}'. "
        "The actual SCADA curve will still be displayed."
    )

# ============================================================
# ANALYZE ALL TURBINES
# ============================================================

with st.spinner(
    "Analyzing turbine power curves..."
):

    results = analyze_all_turbines(
        filtered_scada,
        reference_curve,
        all_uploaded_turbines
    )

st.session_state.results = results

# ============================================================
# SITE HEADER
# ============================================================

st.markdown("---")

total_turbines = len(
    all_uploaded_turbines
)

power_curves = 0
insufficient = 0
no_data = 0
no_curve = 0

for turbine, result in results.items():

    status = result["status"]

    if status in [
        "Normal Performance",
        "Slight Overperformance",
        "High Overperformance",
        "Critical Overperformance",
        "Underperformance",
        "High Underperformance",
        "Critical Underperformance"
    ]:
        power_curves += 1

    elif status == "Insufficient Data":
        insufficient += 1

    elif status == "No Curve":
        no_curve += 1

    else:
        no_data += 1

st.header(
    f"{selected_site} | "
    f"{total_turbines} Turbines | "
    f"{DEFAULT_CAPACITY_KW / 1000:.1f} MW Each | "
    f"Total: {total_turbines * DEFAULT_CAPACITY_KW / 1000:.1f} MW"
)

if start_date is not None:

    st.write(
        f"**Date Range:** "
        f"{start_date} → {end_date}"
    )

# ============================================================
# SUMMARY METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Total Turbines",
        total_turbines
    )

with c2:

    st.metric(
        "Power Curves",
        power_curves
    )

with c3:

    st.metric(
        "Insufficient Data",
        insufficient
    )

with c4:

    st.metric(
        "No Data / Curve",
        no_data + no_curve
    )

# ============================================================
# TURBINE SELECTION
# ============================================================

selected_turbines = all_uploaded_turbines.copy()

if view_mode == "Single Turbine":

    selected_turbine = st.sidebar.selectbox(
        "Select Turbine",
        all_uploaded_turbines
    )

    selected_turbines = [
        selected_turbine
    ]

elif view_mode == "Compare Turbines":

    selected_turbines = st.sidebar.multiselect(
        "Select Turbines",
        all_uploaded_turbines,
        default=all_uploaded_turbines[
            :min(3, len(all_uploaded_turbines))
        ]
    )

    if not selected_turbines:

        st.warning(
            "Please select at least one turbine."
        )

        st.stop()

# ============================================================
# GRAPH SECTION
# ============================================================

st.header("📈 Power Curve Graphs")

if view_mode == "Show All Turbines":

    # --------------------------------------------------------
    # Pagination
    # --------------------------------------------------------

    total_pages = max(
        1,
        math.ceil(
            len(all_uploaded_turbines)
            / GRAPHS_PER_PAGE
        )
    )

    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col1:

        if st.button(
            "⬅ Previous",
            disabled=(
                st.session_state.current_page <= 1
            )
        ):

            st.session_state.current_page -= 1
            st.rerun()

    with col2:

        st.markdown(
            f"<div style='text-align:center;'>"
            f"<b>Page {st.session_state.current_page} "
            f"of {total_pages}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col3:

        if st.button(
            "Next ➡",
            disabled=(
                st.session_state.current_page
                >= total_pages
            )
        ):

            st.session_state.current_page += 1
            st.rerun()

    start_index = (
        st.session_state.current_page - 1
    ) * GRAPHS_PER_PAGE

    end_index = min(
        start_index + GRAPHS_PER_PAGE,
        len(all_uploaded_turbines)
    )

    selected_turbines = (
        all_uploaded_turbines[
            start_index:end_index
        ]
    )

    st.caption(
        f"Showing turbines "
        f"{start_index + 1}–{end_index} "
        f"of {total_turbines} | "
        f"{GRAPHS_PER_PAGE} graphs per page"
    )

# ============================================================
# GRAPH GRID
# ============================================================

for i in range(
    0,
    len(selected_turbines),
    2
):

    row_turbines = selected_turbines[
        i:i + 2
    ]

    columns = st.columns(2)

    for col, turbine in zip(
        columns,
        row_turbines
    ):

        result = results.get(
            turbine
        )

        if result is None:
            continue

        with col:

            st.markdown(
                f"### {turbine}"
            )

            fig = create_power_curve_figure(
                turbine,
                result
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "scrollZoom": True
                }
            )

            dev = result.get(
                "deviation",
                np.nan
            )

            status = result.get(
                "status",
                "No Data"
            )

            if pd.isna(dev):

                dev_text = "N/A"

            else:

                dev_text = (
                    f"{dev:.2f}%"
                )

            if status == "Normal Performance":

                st.success(
                    f"Analysis: Dev: {dev_text} → {status}"
                )

            elif status in [
                "Slight Overperformance",
                "Underperformance",
                "Insufficient Data"
            ]:

                st.warning(
                    f"Analysis: Dev: {dev_text} → {status}"
                )

            elif status in [
                "No Data",
                "No Curve"
            ]:

                st.info(
                    f"Analysis: {status}"
                )

            else:

                st.error(
                    f"Analysis: Dev: {dev_text} → {status}"
                )

            st.caption(
                f"Valid SCADA rows: "
                f"{result.get('valid_rows', 0):,} | "
                f"Standard deviation: "
                f"{result.get('std_power', np.nan):,.2f} kW"
                if not pd.isna(
                    result.get(
                        "std_power",
                        np.nan
                    )
                )
                else
                f"Valid SCADA rows: "
                f"{result.get('valid_rows', 0):,}"
            )

            with st.expander(
                "Show deviation & comment"
            ):

                st.write(
                    f"**Deviation:** {dev_text}"
                )

                st.write(
                    f"**Status:** {status}"
                )

                st.write(
                    f"**Raw rows:** "
                    f"{result.get('raw_rows', 0):,}"
                )

                st.write(
                    f"**Valid rows:** "
                    f"{result.get('valid_rows', 0):,}"
                )

                st.write(
                    f"**Comment:** "
                    f"{result.get('comment', '')}"
                )

# ============================================================
# RANKING
# ============================================================

st.markdown("---")

st.header(
    "🏆 Turbine Ranking / Data Status"
)

ranking_df = create_ranking(
    results
)

if ranking_df.empty:

    st.info(
        "No turbine analysis is available."
    )

else:

    # --------------------------------------------------------
    # Display dataframe
    # --------------------------------------------------------

    display_df = ranking_df.copy()

    display_df["Deviation"] = (
        display_df["Deviation_%"]
        .apply(
            lambda x:
            f"{x:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df["Std Dev"] = (
        display_df["Std Dev (kW)"]
        .apply(
            lambda x:
            f"{x:,.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df = display_df[
        [
            "Turbine",
            "Rank",
            "Deviation",
            "Status",
            "Raw Rows",
            "Valid Rows",
            "Std Dev",
            "Comment"
        ]
    ]

    display_df["Rank"] = (
        display_df["Rank"]
        .apply(
            lambda x:
            int(x)
            if pd.notna(x)
            else "-"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=650
    )

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

st.markdown("---")

st.header(
    "📊 Performance Summary"
)

summary_cols = st.columns(4)

normal_count = 0
over_count = 0
under_count = 0
problem_count = 0

for result in results.values():

    status = result.get(
        "status",
        ""
    )

    if status == "Normal Performance":

        normal_count += 1

    elif "Overperformance" in status:

        over_count += 1

    elif "Underperformance" in status:

        under_count += 1

    elif status in [
        "Insufficient Data",
        "No Curve",
        "No Data"
    ]:

        problem_count += 1

with summary_cols[0]:

    st.metric(
        "Normal",
        normal_count
    )

with summary_cols[1]:

    st.metric(
        "Overperformance",
        over_count
    )

with summary_cols[2]:

    st.metric(
        "Underperformance",
        under_count
    )

with summary_cols[3]:

    st.metric(
        "Data Issues",
        problem_count
    )

# ============================================================
# REFERENCE CURVE PREVIEW
# ============================================================

with st.expander(
    "🔎 Reference Curve Preview"
):

    if reference_curve is not None:

        st.dataframe(
            reference_curve,
            use_container_width=True,
            hide_index=True
        )

        fig_ref = go.Figure()

        fig_ref.add_trace(
            go.Scatter(
                x=reference_curve["Wind"],
                y=reference_curve["ReferencePower"],
                mode="lines+markers",
                name="Reference Curve",
                line=dict(
                    color="red",
                    width=3
                )
            )
        )

        fig_ref.update_layout(
            height=450,
            xaxis_title="Wind Speed (m/s)",
            yaxis_title="Reference Power (kW)",
            xaxis=dict(
                range=[2.5, 16]
            )
        )

        st.plotly_chart(
            fig_ref,
            use_container_width=True
        )

    else:

        st.warning(
            "Reference curve is not available."
        )

# ============================================================
# DATA INFORMATION
# ============================================================

with st.expander(
    "ℹ️ Data Information"
):

    st.write(
        f"**Uploaded SCADA rows:** "
        f"{len(scada_df):,}"
    )

    st.write(
        f"**Filtered SCADA rows:** "
        f"{len(filtered_scada):,}"
    )

    st.write(
        f"**Detected turbines:** "
        f"{len(all_uploaded_turbines)}"
    )

    st.write(
        f"**Selected site:** "
        f"{selected_site}"
    )

    st.write(
        f"**Reference file:** "
        f"{st.session_state.reference_name}"
    )

    if "_Time" in scada_df.columns:

        valid_times = scada_df[
            "_Time"
        ].dropna()

        if not valid_times.empty:

            st.write(
                f"**Available SCADA period:** "
                f"{valid_times.min()} → "
                f"{valid_times.max()}"
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Power Curve Analytics Dashboard | "
    "SCADA vs Reference Power Curve"
)
