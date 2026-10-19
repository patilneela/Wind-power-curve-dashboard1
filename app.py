import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi-Site Power Curve Analytics",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================
BIN_WIDTH = 0.5

# These are the standard SCADA columns you said will be same for all sites.
WTG_COL = "Name"
TIME_COL = "Time"
STATUS_COL = "PCurveStsAve"
WIND_SPEED_COL = "WindSpeedAve"
WIND_STD_COL = "WindSpeedStd"
PITCH_STD_COL = "BladePitchStd"
POWER_COL = "ActivePWAve"

# Optional site column candidates in uploaded SCADA file.
SITE_COL_CANDIDATES = [
    "Site",
    "SiteName",
    "Plant",
    "WindFarm",
    "Project",
    "Location"
]

# ============================================================
# SMALL HELPERS
# ============================================================
def normalize_text(s):
    """
    Normalize text so column matching becomes easier.
    Example:
    'Site Name', 'site_name', 'SITE-NAME' -> 'sitename'
    """
    if pd.isna(s):
        return ""
    return (
        str(s).strip().lower()
        .replace("-", "")
        .replace("_", "")
        .replace(",", "")
        .replace(" ", "")
        .replace("\n", "")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
    )


def detect_column(columns, candidates):
    """
    Detect the best matching column from a list of candidate names.
    Used to auto-detect the site column from the SCADA file.
    """
    normalized_map = {c: normalize_text(c) for c in columns}

    # Exact normalized match
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        exact = [col for col, norm in normalized_map.items() if norm == candidate_norm]
        if exact:
            return exact[0]

    # Partial contains match
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        contains = [col for col, norm in normalized_map.items() if candidate_norm in norm]
        if contains:
            return contains[0]

    return None


def compute_preset_range(preset: str, today_ts: pd.Timestamp):
    """
    Return start date and end date for a preset date range.
    """
    today_date = today_ts.normalize().date()

    if preset == "Today":
        return today_date, today_date

    if preset == "This Week":
        monday = today_date - timedelta(days=today_ts.weekday())
        return monday, today_date

    if preset == "Last Week":
        this_monday = today_date - timedelta(days=today_ts.weekday())
        start = this_monday - timedelta(days=7)
        end = this_monday - timedelta(days=1)
        return start, end

    if preset == "This Month":
        start = today_date.replace(day=1)
        return start, today_date

    if preset == "Last Month":
        first_this_month = today_date.replace(day=1)
        last_month_end = first_this_month - timedelta(days=1)
        start = last_month_end.replace(day=1)
        return start, last_month_end

    return None


def generate_comment(dev):
    """
    Convert average deviation % into a human-readable performance comment.
    """
    if dev is None or pd.isna(dev):
        return "Deviation not available"

    dev = round(dev, 2)

    if dev < -72:
        return f"Dev: {dev}% → Extreme issue (Data unreliable)"
    elif dev < -8:
        return f"Dev: {dev}% → Severe underperformance (Blade/Dust/Yaw issue)"
    elif dev < -2:
        return f"Dev: {dev}% → Underperformance"
    elif dev > 72:
        return f"Dev: {dev}% → Abnormal high (Sensor/Data issue)"
    elif dev > 8:
        return f"Dev: {dev}% → High overperformance"
    elif dev > 2:
        return f"Dev: {dev}% → Slight overperformance"
    else:
        return f"Dev: {dev}% → Normal performance"


# ============================================================
# LOAD SCADA
# ============================================================
@st.cache_data(show_spinner=True)
def load_scada(uploaded_file):
    """
    Load uploaded SCADA CSV file.
    Using chunksize helps handle bigger files.
    """
    chunks = pd.read_csv(uploaded_file, chunksize=200000, low_memory=False)
    df_local = pd.concat(chunks, ignore_index=True)
    df_local.columns = df_local.columns.str.strip()
    return df_local


# ============================================================
# LOAD REFERENCE EXCEL
# ============================================================
@st.cache_data(show_spinner=True)
def load_reference_excel(uploaded_file):
    """
    Read the reference Excel file.

    Expected format:
    - Sheet name: Power Curve
    - Row index 2 (3rd row) contains site names in columns B onward
    - Row index 3 onward contains wind speed in column A
    - Row index 3 onward contains corresponding reference power in each site column
    """
    ref_raw = pd.read_excel(uploaded_file, sheet_name="Power Curve", header=None)
    site_map = build_site_reference_map(ref_raw)

    if not site_map:
        raise ValueError("No valid site reference curves found in 'Power Curve' sheet.")

    return site_map


def build_site_reference_map(ref_raw):
    """
    Build a dictionary:
    {
        'Site A': DataFrame(ReferenceWindSpeed_mps, ReferencePower_kW),
        'Site B': DataFrame(...),
        ...
    }
    """
    site_headers = ref_raw.iloc[2].copy()
    wind_series = pd.to_numeric(ref_raw.iloc[3:, 0], errors="coerce")

    site_map = {}

    for col_idx in range(1, ref_raw.shape[1]):
        site_name = str(site_headers.iloc[col_idx]).strip()

        if not site_name or site_name.lower() in {"nan", "none", "site", "site name"}:
            continue

        ref_power = pd.to_numeric(ref_raw.iloc[3:, col_idx], errors="coerce")

        ref = pd.DataFrame({
            "ReferenceWindSpeed_mps": wind_series,
            "ReferencePower_kW": ref_power
        }).dropna()

        ref = (
            ref.sort_values("ReferenceWindSpeed_mps")
            .drop_duplicates("ReferenceWindSpeed_mps")
            .reset_index(drop=True)
        )

        if len(ref) < 2:
            continue

        site_map[site_name] = ref

    return site_map


# ============================================================
# DATA VALIDATION
# ============================================================
def validate_scada_columns(df):
    """
    Make sure the uploaded SCADA file has all mandatory columns.
    """
    required = [
        WTG_COL,
        TIME_COL,
        STATUS_COL,
        WIND_SPEED_COL,
        WIND_STD_COL,
        PITCH_STD_COL,
        POWER_COL
    ]

    missing = [c for c in required if c not in df.columns]
    return missing


def prepare_scada(df):
    """
    Convert SCADA columns into correct types.
    """
    df = df.copy()

    df[WTG_COL] = df[WTG_COL].astype(str).str.strip()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    numeric_cols = [
        STATUS_COL,
        WIND_SPEED_COL,
        WIND_STD_COL,
        PITCH_STD_COL,
        POWER_COL
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[WTG_COL, TIME_COL])
    return df


# ============================================================
# SITE COLUMN HANDLING
# ============================================================
def ensure_site_column(df, ref_site_names):
    """
    Ensure a usable site column is present in the SCADA data.

    Cases:
    1. Site column exists -> use it.
    2. No site column exists and only one reference site exists -> assign that site to all rows.
    3. No site column exists and multiple reference sites exist -> user must choose mapping later.
    """
    df = df.copy()
    detected_site_col = detect_column(df.columns, SITE_COL_CANDIDATES)

    if detected_site_col:
        df["__Site__"] = df[detected_site_col].astype(str).str.strip()
        return df, detected_site_col, None

    if len(ref_site_names) == 1:
        only_site = ref_site_names[0]
        df["__Site__"] = only_site
        return df, None, f"No site column found in SCADA. Assigned all rows to '{only_site}'."

    return df, None, (
        "No site column found in SCADA file. "
        "Since the reference file has multiple sites, please upload a SCADA file "
        "containing a site column such as Site / SiteName / Plant / WindFarm."
    )


# ============================================================
# BASE FILTERS
# ============================================================
def build_base_df(df_in, wind_min, wind_max):
    """
    Apply base filters before power curve comparison:
    - non-null wind speed
    - non-null power
    - wind speed within user selected range
    - power >= 0
    """
    df_out = df_in.dropna(subset=[WIND_SPEED_COL, POWER_COL]).copy()

    if df_out.empty:
        return df_out

    df_out = df_out[
        (df_out[WIND_SPEED_COL] >= wind_min)
        & (df_out[WIND_SPEED_COL] <= wind_max)
        & (df_out[POWER_COL] >= 0)
    ].copy()

    return df_out


# ============================================================
# CORE POWER CURVE ANALYSIS FOR ONE TURBINE
# ============================================================
def apply_power_curve_logic(df_in, ref_curve_raw):
    """
    Apply the exact logic from your original script:
    - PCurveStsAve == 100
    - WindSpeedStd < 1
    - -5 <= BladePitchStd <= 19
    - keep only rows within reference curve wind range
    - interpolate expected power
    - compute deviation metrics
    - bin by 0.5 m/s
    """
    merged_empty = pd.DataFrame(columns=["WindBin", "RefPower", "AvgPower", "Deviation_%"])

    if df_in.empty:
        return {
            "power_curve_df": df_in.iloc[0:0].copy(),
            "merged": merged_empty,
            "avg_dev": None,
            "deviation_filter_rows": 0
        }

    mask_status = df_in[STATUS_COL].eq(100)
    mask_wind_std = df_in[WIND_STD_COL].lt(1)
    mask_pitch_std = df_in[PITCH_STD_COL].between(-5, 19, inclusive="both")

    power_curve_df = df_in.loc[mask_status & mask_wind_std & mask_pitch_std].copy()

    ref_min = ref_curve_raw["ReferenceWindSpeed_mps"].min()
    ref_max = ref_curve_raw["ReferenceWindSpeed_mps"].max()

    power_curve_df = power_curve_df[
        power_curve_df[WIND_SPEED_COL].between(ref_min, ref_max, inclusive="both")
    ].copy()

    if power_curve_df.empty:
        ref_plot = ref_curve_raw.rename(columns={
            "ReferenceWindSpeed_mps": "WindBin",
            "ReferencePower_kW": "RefPower"
        }).copy()
        ref_plot["AvgPower"] = np.nan
        ref_plot["Deviation_%"] = np.nan

        return {
            "power_curve_df": power_curve_df,
            "merged": ref_plot,
            "avg_dev": None,
            "deviation_filter_rows": 0
        }

    # Interpolate reference power
    power_curve_df["ReferencePower_kW"] = np.interp(
        power_curve_df[WIND_SPEED_COL].to_numpy(dtype=float),
        ref_curve_raw["ReferenceWindSpeed_mps"].to_numpy(dtype=float),
        ref_curve_raw["ReferencePower_kW"].to_numpy(dtype=float)
    )

    # Deviation calculations
    power_curve_df["Deviation_kW"] = (
        power_curve_df[POWER_COL] - power_curve_df["ReferencePower_kW"]
    )
    power_curve_df["Shortfall_kW"] = -power_curve_df["Deviation_kW"]
    power_curve_df["Deviation_pct"] = np.where(
        power_curve_df["ReferencePower_kW"] > 0,
        100 * power_curve_df["Deviation_kW"] / power_curve_df["ReferencePower_kW"],
        np.nan
    )

    # Bin wind speed
    power_curve_df["WindSpeedBin_mps"] = (
        np.floor(power_curve_df[WIND_SPEED_COL] / BIN_WIDTH) * BIN_WIDTH
    ).round(6)

    actual = (
        power_curve_df.groupby("WindSpeedBin_mps", as_index=False)
        .agg(
            AvgPower=(POWER_COL, "mean"),
            MeanReferencePower=("ReferencePower_kW", "mean"),
            Deviation_pct=("Deviation_pct", "mean")
        )
        .rename(columns={"WindSpeedBin_mps": "WindBin"})
    )

    ref_plot = ref_curve_raw.copy()
    ref_plot["WindBin"] = ref_plot["ReferenceWindSpeed_mps"]
    ref_plot["RefPower"] = ref_plot["ReferencePower_kW"]
    ref_plot = ref_plot[["WindBin", "RefPower"]]

    merged = ref_plot.merge(actual, on="WindBin", how="left")
    merged["Deviation_%"] = merged["Deviation_pct"]
    merged = merged.drop(columns=["Deviation_pct"], errors="ignore")

    mean_actual = power_curve_df[POWER_COL].mean()
    mean_reference = power_curve_df["ReferencePower_kW"].mean()

    if pd.notna(mean_reference) and mean_reference > 0:
        avg_dev = 100 * (mean_actual - mean_reference) / mean_reference
    else:
        avg_dev = None

    if pd.isna(avg_dev):
        avg_dev = None

    return {
        "power_curve_df": power_curve_df,
        "merged": merged,
        "avg_dev": avg_dev,
        "deviation_filter_rows": len(power_curve_df)
    }


# ============================================================
# PROCESS ONE TURBINE
# ============================================================
def process_turbine(df_site, turbine_name, ref_curve_raw, wind_min, wind_max):
    """
    Build all required turbine-level outputs.
    """
    df_t = df_site[df_site[WTG_COL] == turbine_name].copy()
    raw_rows = len(df_t)

    result = {
        "turbine": turbine_name,
        "raw_rows": raw_rows,
        "status": "No Data",
        "comment": "No data available",
        "base_df": pd.DataFrame(),
        "power_curve_df": pd.DataFrame(),
        "merged": pd.DataFrame(columns=["WindBin", "RefPower", "AvgPower", "Deviation_%"]),
        "avg_dev": None,
        "base_power_sample_count": 0,
        "deviation_filter_rows": 0,
    }

    if raw_rows == 0:
        return result

    base_df = build_base_df(df_t, wind_min, wind_max)

    if base_df.empty:
        result["status"] = "Filtered Out"
        result["comment"] = "All rows removed by base filters"
        return result

    result["base_df"] = base_df
    result["base_power_sample_count"] = len(base_df)

    pc_result = apply_power_curve_logic(base_df, ref_curve_raw)

    result["power_curve_df"] = pc_result["power_curve_df"]
    result["merged"] = pc_result["merged"]
    result["avg_dev"] = pc_result["avg_dev"]
    result["deviation_filter_rows"] = pc_result["deviation_filter_rows"]

    if result["avg_dev"] is None:
        result["status"] = "Low Data"
        if result["power_curve_df"].empty:
            result["comment"] = (
                "No rows passed required filters "
                "(PCurveStsAve=100, WindSpeedStd<1, -5<=BladePitchStd<=19)"
            )
        else:
            result["comment"] = "Insufficient overlap with reference curve"
    else:
        result["status"] = "OK"
        result["comment"] = generate_comment(result["avg_dev"])

    return result


# ============================================================
# BUILD SITE ANALYSIS
# ============================================================
def build_site_analysis(df_site, processed_results, ref_curve_raw, site_name, start_day, end_day):
    """
    Build site-level summary tables:
    - filter counts
    - overall summary
    - WTG summary
    - fleet wind bins
    - WTG wind bins
    - filtered SCADA details
    """
    analysis_frames = [
        r["power_curve_df"].copy()
        for r in processed_results.values()
        if isinstance(r.get("power_curve_df"), pd.DataFrame) and not r["power_curve_df"].empty
    ]

    if analysis_frames:
        analysis_data = pd.concat(analysis_frames, ignore_index=True)
    else:
        analysis_data = pd.DataFrame()

    mask_status = df_site[STATUS_COL].eq(100)
    mask_wind_std = df_site[WIND_STD_COL].lt(1)
    mask_pitch_std = df_site[PITCH_STD_COL].between(-5, 19, inclusive="both")

    filter_counts_df = pd.DataFrame([
        ["Total SCADA records in selected date range", len(df_site)],
        ["PCurveStsAve = 100", int(mask_status.sum())],
        ["After PCurveStsAve = 100 and WindSpeedStd < 1", int((mask_status & mask_wind_std).sum())],
        ["After all three required filters", int((mask_status & mask_wind_std & mask_pitch_std).sum())],
        ["Compared to reference curve", len(analysis_data)],
    ], columns=["Filter Stage", "Record Count"])

    if analysis_data.empty:
        return {
            "analysis_data": analysis_data,
            "filter_counts_df": filter_counts_df,
            "overall_df": None,
            "wtg_summary_df": None,
            "fleet_bins_df": None,
            "wtg_bins_df": None
        }

    analysis_data = analysis_data.copy()
    analysis_data["WTG"] = analysis_data[WTG_COL].astype(str)
    analysis_data["DateTime"] = analysis_data[TIME_COL]
    analysis_data["WindSpeed_mps"] = analysis_data[WIND_SPEED_COL]
    analysis_data["ActualPower_kW"] = analysis_data[POWER_COL]

    # WTG wind bins
    wtg_bins_df = (
        analysis_data.groupby(["WTG", "WindSpeedBin_mps"], as_index=False)
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
    )

    # Fleet bins
    fleet_bins_df = (
        analysis_data.groupby("WindSpeedBin_mps", as_index=False)
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
        )
    )

    # WTG summary
    wtg_summary_df = (
        analysis_data.groupby("WTG", as_index=False)
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
    )

    wtg_summary_df["AggregateDeviation_pct"] = np.where(
        wtg_summary_df["MeanReferencePower_kW"] > 0,
        100 * (
            wtg_summary_df["MeanActualPower_kW"] - wtg_summary_df["MeanReferencePower_kW"]
        ) / wtg_summary_df["MeanReferencePower_kW"],
        np.nan
    )

    wtg_summary_df = wtg_summary_df.sort_values("AggregateDeviation_pct", na_position="last")

    ref_min = ref_curve_raw["ReferenceWindSpeed_mps"].min()
    ref_max = ref_curve_raw["ReferenceWindSpeed_mps"].max()

    overall_df = pd.DataFrame([
        ["Selected site", site_name],
        ["Selected date range", f"{start_day} to {end_day}"],
        ["Total turbines in site", int(df_site[WTG_COL].nunique())],
        ["Turbines compared", int(analysis_data["WTG"].nunique())],
        ["Records compared to reference curve", int(len(analysis_data))],
        ["Reference curve min wind speed (m/s)", ref_min],
        ["Reference curve max wind speed (m/s)", ref_max],
        ["Filter 1", "PCurveStsAve = 100"],
        ["Filter 2", "WindSpeedStd < 1"],
        ["Filter 3", "-5 <= BladePitchStd <= 19"],
        ["Wind-speed bin width (m/s)", BIN_WIDTH],
        ["Deviation sign", "Actual power - reference power; negative means below reference"],
    ], columns=["Metric", "Value"])

    return {
        "analysis_data": analysis_data,
        "filter_counts_df": filter_counts_df,
        "overall_df": overall_df,
        "wtg_summary_df": wtg_summary_df,
        "fleet_bins_df": fleet_bins_df,
        "wtg_bins_df": wtg_bins_df
    }


# ============================================================
# PLOT HELPERS
# ============================================================
def plot_turbine_power_curve(power_curve_df, merged, title):
    """
    Plot one turbine power curve with:
    - reference line
    - filtered SCADA scatter
    - actual binned average line
    """
    fig = go.Figure()

    if merged is not None and not merged.empty:
        fig.add_trace(
            go.Scatter(
                x=merged["WindBin"],
                y=merged["RefPower"],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Reference Power Curve"
            )
        )

    if power_curve_df is not None and not power_curve_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=power_curve_df[WIND_SPEED_COL],
                y=power_curve_df[POWER_COL],
                mode="markers",
                marker=dict(size=4, color="rgba(30,144,255,0.18)"),
                name="Filtered SCADA"
            )
        )

    avg_curve = pd.DataFrame()
    if merged is not None and not merged.empty:
        avg_curve = merged.dropna(subset=["AvgPower"]).copy()

    if not avg_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=avg_curve["WindBin"],
                y=avg_curve["AvgPower"],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=5),
                name="SCADA mean by 0.5 m/s bin"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=320,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(font=dict(size=9))
    )

    return fig


def plot_site_power_curve(ref_curve_raw, analysis_data, fleet_bins_df, site_name):
    """
    Plot combined site-level power curve.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ref_curve_raw["ReferenceWindSpeed_mps"],
        y=ref_curve_raw["ReferencePower_kW"],
        mode="lines",
        line=dict(color="red", width=3, dash="dash"),
        name="Reference Power Curve"
    ))

    fig.add_trace(go.Scattergl(
        x=analysis_data["WindSpeed_mps"],
        y=analysis_data["ActualPower_kW"],
        mode="markers",
        marker=dict(size=5, color="rgba(30,144,255,0.14)"),
        name="Filtered SCADA"
    ))

    fig.add_trace(go.Scatter(
        x=fleet_bins_df["MeanWindSpeed_mps"],
        y=fleet_bins_df["MeanActualPower_kW"],
        mode="lines+markers",
        line=dict(color="blue", width=3),
        marker=dict(size=7),
        name="Fleet mean by 0.5 m/s bin"
    ))

    fig.update_layout(
        title=f"{site_name} - Site Level Power Curve Analysis",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=600,
        template="plotly_white"
    )

    return fig


def plot_site_deviation(fleet_bins_df, site_name):
    """
    Plot site-level mean deviation by wind-speed bin.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line_width=1, line_color="gray")

    fig.add_trace(go.Scatter(
        x=fleet_bins_df["MeanWindSpeed_mps"],
        y=fleet_bins_df["MeanDeviation_kW"],
        mode="lines+markers",
        line=dict(color="blue", width=3),
        marker=dict(size=7),
        name="Mean deviation"
    ))

    fig.update_layout(
        title=f"{site_name} - Mean Deviation by Wind-Speed Bin",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Mean Deviation (Actual - Reference, kW)",
        height=400,
        template="plotly_white"
    )

    return fig


# ============================================================
# EXCEL EXPORT
# ============================================================
def build_excel_download(site_analysis, ref_curve_raw):
    """
    Create in-memory Excel file for download.
    """
    excel_buffer = io.BytesIO()

    analysis_data = site_analysis["analysis_data"]
    filter_counts_df = site_analysis["filter_counts_df"]
    overall_df = site_analysis["overall_df"]
    wtg_summary_df = site_analysis["wtg_summary_df"]
    fleet_bins_df = site_analysis["fleet_bins_df"]
    wtg_bins_df = site_analysis["wtg_bins_df"]

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        if overall_df is not None:
            overall_df.to_excel(writer, sheet_name="Overall Summary", index=False)

        if filter_counts_df is not None:
            filter_counts_df.to_excel(writer, sheet_name="Filter Counts", index=False)

        if wtg_summary_df is not None:
            wtg_summary_df.to_excel(writer, sheet_name="WTG Summary", index=False)

        if fleet_bins_df is not None:
            fleet_bins_df.to_excel(writer, sheet_name="Fleet Wind Bins", index=False)

        if wtg_bins_df is not None:
            wtg_bins_df.to_excel(writer, sheet_name="WTG Wind Bins", index=False)

        if analysis_data is not None and not analysis_data.empty:
            detail_cols = [
                "WTG",
                "DateTime",
                STATUS_COL,
                WIND_STD_COL,
                PITCH_STD_COL,
                "WindSpeed_mps",
                "ActualPower_kW",
                "ReferencePower_kW",
                "Deviation_kW",
                "Shortfall_kW",
                "Deviation_pct",
                "WindSpeedBin_mps",
            ]
            detail_cols = [c for c in detail_cols if c in analysis_data.columns]
            analysis_data[detail_cols].to_excel(
                writer,
                sheet_name="Filtered SCADA Details",
                index=False
            )

        ref_curve_raw.to_excel(writer, sheet_name="Reference Curve", index=False)

    excel_buffer.seek(0)
    return excel_buffer


# ============================================================
# UI HEADER
# ============================================================
st.title("Multi-Site Power Curve Analytics Dashboard")
st.caption(
    "Upload one SCADA CSV and one reference Excel. "
    "The app will analyze all sites and display all turbines as mini-charts."
)

# ============================================================
# SIDEBAR UPLOADS
# ============================================================
st.sidebar.header("Upload Files")

scada_file = st.sidebar.file_uploader(
    "Upload SCADA CSV",
    type=["csv"]
)

ref_file = st.sidebar.file_uploader(
    "Upload Reference Excel",
    type=["xlsx"]
)

if scada_file is None:
    st.info("Please upload SCADA CSV file to continue.")
    st.stop()

if ref_file is None:
    st.info("Please upload reference Excel file to continue.")
    st.stop()

# ============================================================
# LOAD FILES
# ============================================================
with st.spinner("Loading SCADA data..."):
    df = load_scada(scada_file)

missing_cols = validate_scada_columns(df)
if missing_cols:
    st.error("SCADA file is missing required columns:")
    st.write(missing_cols)
    st.stop()

df = prepare_scada(df)

if df.empty:
    st.error("SCADA file contains no usable rows after parsing.")
    st.stop()

try:
    site_ref_map = load_reference_excel(ref_file)
except Exception as e:
    st.error("Unable to load reference Excel.")
    st.code(str(e))
    st.stop()

ref_site_names = list(site_ref_map.keys())

df, detected_site_col, site_error = ensure_site_column(df, ref_site_names)

if site_error:
    st.error(site_error)
    st.stop()

if detected_site_col:
    st.sidebar.success(f"Detected site column: {detected_site_col}")
else:
    st.sidebar.info("No site column detected; all rows mapped to the only reference site.")

# ============================================================
# DATE FILTER
# ============================================================
st.sidebar.header("Date Range")

max_ts = df[TIME_COL].max()
base_date = max_ts.normalize().date() if pd.notna(max_ts) else pd.Timestamp.today().date()

default_start = base_date - timedelta(days=30)
default_end = base_date

date_option = st.sidebar.selectbox(
    "Date Option",
    ["Manual", "Today", "This Week", "This Month", "Last Week", "Last Month"]
)

if date_option == "Manual":
    start_day = st.sidebar.date_input("Start Date", value=default_start)
    end_day = st.sidebar.date_input("End Date", value=default_end)
else:
    rng = compute_preset_range(date_option, pd.Timestamp.today())
    start_day, end_day = rng if rng else (default_start, default_end)

df["_date_only"] = df[TIME_COL].dt.date
df = df[(df["_date_only"] >= start_day) & (df["_date_only"] <= end_day)].copy()
df = df.drop(columns=["_date_only"])

if df.empty:
    st.warning("No SCADA data available for selected date range.")
    st.stop()

# ============================================================
# OTHER FILTERS
# ============================================================
st.sidebar.header("Analysis Filters")

wind_min = st.sidebar.number_input("Wind Min (m/s)", value=3.0)
wind_max = st.sidebar.number_input("Wind Max (m/s)", value=15.0)

available_sites_in_scada = sorted(df["__Site__"].dropna().unique())

selected_sites = st.sidebar.multiselect(
    "Select Sites",
    available_sites_in_scada,
    default=available_sites_in_scada
)

if not selected_sites:
    st.warning("Please select at least one site.")
    st.stop()

# ============================================================
# SHOW LOGIC
# ============================================================
with st.expander("Applied Formula and Filters", expanded=False):
    st.markdown("### Required Filters")
    st.code(
        "\n".join([
            f"{STATUS_COL} == 100",
            f"{WIND_STD_COL} < 1",
            f"-5 <= {PITCH_STD_COL} <= 19",
        ]),
        language="python"
    )

    st.markdown("### Base Filters")
    st.code(
        "\n".join([
            f"{WIND_SPEED_COL} is not null",
            f"{POWER_COL} is not null",
            f"{wind_min} <= {WIND_SPEED_COL} <= {wind_max}",
            f"{POWER_COL} >= 0",
        ]),
        language="python"
    )

    st.markdown("### Deviation Formula")
    st.code(
        "\n".join([
            "ReferencePower_kW = linear interpolation from selected site reference curve",
            "Deviation_kW = ActualPower_kW - ReferencePower_kW",
            "Shortfall_kW = -Deviation_kW",
            "Deviation_pct = 100 * Deviation_kW / ReferencePower_kW",
        ]),
        language="python"
    )

# ============================================================
# MAIN LOOP: PROCESS EACH SITE
# ============================================================
all_site_summaries = []

for site_name in selected_sites:
    st.markdown("---")
    st.header(f"Site: {site_name}")

    if site_name not in site_ref_map:
        st.warning(f"No reference curve found for site: {site_name}")
        continue

    ref_curve_raw = site_ref_map[site_name].copy()
    df_site = df[df["__Site__"] == site_name].copy()

    if df_site.empty:
        st.info(f"No SCADA rows found for {site_name} in selected date range.")
        continue

    turbines = sorted(df_site[WTG_COL].dropna().unique())

    st.write(f"**Turbines in site:** {len(turbines)}")
    st.write(f"**SCADA rows in selected date range:** {len(df_site):,}")

    processed_results = {}
    ranking_rows = []

    # --------------------------------------------------------
    # MINI CHARTS FOR ALL TURBINES
    # --------------------------------------------------------
    st.subheader("All Turbine Mini-Charts")

    cols_per_row = 2

    for i in range(0, len(turbines), cols_per_row):
        row_turbines = turbines[i:i + cols_per_row]
        cols = st.columns(cols_per_row)

        for j, turbine_name in enumerate(row_turbines):
            res = process_turbine(df_site, turbine_name, ref_curve_raw, wind_min, wind_max)
            processed_results[turbine_name] = res

            ranking_rows.append({
                "Site": site_name,
                "Turbine": turbine_name,
                "Deviation_%": None if res["avg_dev"] is None else round(res["avg_dev"], 2),
                "Base_Power_Samples": res["base_power_sample_count"],
                "Deviation_Filtered_Samples": res["deviation_filter_rows"],
                "Status": res["status"],
                "Rows": res["raw_rows"],
                "Comment": res["comment"],
            })

            with cols[j]:
                fig = plot_turbine_power_curve(
                    res["power_curve_df"],
                    res["merged"],
                    f"{turbine_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Deviation: {'NA' if res['avg_dev'] is None else round(res['avg_dev'], 2)}%")
                st.caption(f"Base Samples: {res['base_power_sample_count']}")
                st.caption(f"Deviation Samples: {res['deviation_filter_rows']}")
                st.caption(res["comment"])

    # --------------------------------------------------------
    # RANKING TABLE
    # --------------------------------------------------------
    st.subheader("Turbine Ranking")
    ranking_df = pd.DataFrame(ranking_rows)

    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(by="Deviation_%", na_position="last")
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # SITE-LEVEL ANALYSIS
    # --------------------------------------------------------
    site_analysis = build_site_analysis(
        df_site=df_site,
        processed_results=processed_results,
        ref_curve_raw=ref_curve_raw,
        site_name=site_name,
        start_day=start_day,
        end_day=end_day,
    )

    analysis_data = site_analysis["analysis_data"]
    filter_counts_df = site_analysis["filter_counts_df"]
    overall_df = site_analysis["overall_df"]
    wtg_summary_df = site_analysis["wtg_summary_df"]
    fleet_bins_df = site_analysis["fleet_bins_df"]
    wtg_bins_df = site_analysis["wtg_bins_df"]

    if analysis_data is not None and not analysis_data.empty:
        st.subheader("Site-Level Power Curve")
        site_fig = plot_site_power_curve(ref_curve_raw, analysis_data, fleet_bins_df, site_name)
        st.plotly_chart(site_fig, use_container_width=True)

        st.subheader("Site-Level Deviation by Wind-Speed Bin")
        dev_fig = plot_site_deviation(fleet_bins_df, site_name)
        st.plotly_chart(dev_fig, use_container_width=True)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Overall Summary",
            "Filter Counts",
            "WTG Summary",
            "Fleet Wind Bins",
            "WTG Wind Bins",
            "Filtered SCADA Details"
        ])

        with tab1:
            st.dataframe(overall_df, use_container_width=True, hide_index=True)

        with tab2:
            st.dataframe(filter_counts_df, use_container_width=True, hide_index=True)

        with tab3:
            st.dataframe(wtg_summary_df, use_container_width=True, hide_index=True)

        with tab4:
            st.dataframe(fleet_bins_df, use_container_width=True, hide_index=True)

        with tab5:
            st.dataframe(wtg_bins_df, use_container_width=True, hide_index=True)

        with tab6:
            detail_cols = [
                "WTG",
                "DateTime",
                STATUS_COL,
                WIND_STD_COL,
                PITCH_STD_COL,
                "WindSpeed_mps",
                "ActualPower_kW",
                "ReferencePower_kW",
                "Deviation_kW",
                "Shortfall_kW",
                "Deviation_pct",
                "WindSpeedBin_mps",
            ]
            detail_cols = [c for c in detail_cols if c in analysis_data.columns]
            st.dataframe(analysis_data[detail_cols], use_container_width=True, hide_index=True)

        # Excel download for this site
        excel_buffer = build_excel_download(site_analysis, ref_curve_raw)

        st.download_button(
            label=f"Download Excel Report - {site_name}",
            data=excel_buffer.getvalue(),
            file_name=f"{site_name.replace(' ', '_')}_Power_Curve_Analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        all_site_summaries.append({
            "Site": site_name,
            "SCADA_Rows": len(df_site),
            "Turbines": df_site[WTG_COL].nunique(),
            "Compared_Rows": len(analysis_data),
            "Mean_Deviation_pct": round(wtg_summary_df["AggregateDeviation_pct"].mean(), 2)
            if wtg_summary_df is not None and not wtg_summary_df.empty else np.nan
        })

    else:
        st.info(
            f"No SCADA samples passed the required filters and reference range for site: {site_name}"
        )
        st.dataframe(filter_counts_df, use_container_width=True, hide_index=True)

# ============================================================
# MULTI-SITE OVERALL SUMMARY
# ============================================================
if all_site_summaries:
    st.markdown("---")
    st.header("Multi-Site Summary")

    all_site_summary_df = pd.DataFrame(all_site_summaries)
    st.dataframe(all_site_summary_df, use_container_width=True, hide_index=True)

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        x=all_site_summary_df["Site"],
        y=all_site_summary_df["Mean_Deviation_pct"],
        text=all_site_summary_df["Mean_Deviation_pct"],
        textposition="outside"
    ))
    fig_compare.update_layout(
        title="Average Aggregate Deviation by Site",
        xaxis_title="Site",
        yaxis_title="Mean Aggregate Deviation (%)",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_compare, use_container_width=True)
