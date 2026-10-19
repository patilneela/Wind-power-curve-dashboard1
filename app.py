import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import os
import io

from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")

# =========================
# LOGIN
# =========================
def login_gate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("Login Required")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

    if submitted:
        cfg = st.secrets.get("auth", {})
        ok = (username == cfg.get("username")) and (password == cfg.get("password"))
        if ok:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()


login_gate()

if st.sidebar.button("Logout", key="logout_btn"):
    st.session_state.authenticated = False
    st.rerun()

# =========================
# SAFE KALEIDO CHECK
# =========================
try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, "Envision.png")
BIN_SIZE = 0.5

# =========================
# HELPERS
# =========================
def normalize_text(s):
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
    normalized_map = {c: normalize_text(c) for c in columns}

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        exact = [col for col, norm in normalized_map.items() if norm == candidate_norm]
        if exact:
            return exact[0]

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        contains = [col for col, norm in normalized_map.items() if candidate_norm in norm]
        if contains:
            return contains[0]

    return None


def compute_preset_range(preset: str, today_ts: pd.Timestamp):
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


def build_metric_curve(df_t, wind_col, metric_col):
    tmp = df_t[[wind_col, metric_col]].dropna().copy()

    if tmp.empty:
        return pd.DataFrame(columns=["WindBin", "AvgMetric"])

    tmp["WindBin"] = (np.floor(tmp[wind_col] / BIN_SIZE) * BIN_SIZE).round(6)
    out = tmp.groupby("WindBin", as_index=False).agg(AvgMetric=(metric_col, "mean"))
    return out


def get_nacelle_band(avg_nacelle):
    if avg_nacelle is None or pd.isna(avg_nacelle):
        return "NA"
    if 0 <= avg_nacelle < 60:
        return "0-60"
    elif 60 <= avg_nacelle < 120:
        return "60-120"
    elif 120 <= avg_nacelle < 180:
        return "120-180"
    elif 180 <= avg_nacelle < 240:
        return "180-240"
    elif 240 <= avg_nacelle <= 360:
        return "240-360"
    return "Out of Range"


def generate_nacelle_comment(avg_nacelle):
    if avg_nacelle is None or pd.isna(avg_nacelle):
        return "Nacelle average not available"
    band = get_nacelle_band(avg_nacelle)
    val = round(avg_nacelle, 2)
    if band == "Out of Range":
        return f"NacelleAvg: {val}° → Out of configured range"
    return f"NacelleAvg: {val}° → Band {band}"


def generate_comment(dev):
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


def build_dashboard_filtered_df(
    df_in,
    wind_col,
    power_col,
    pitch_col,
    nacelle_col,
    wind_min,
    wind_max,
    pitch_min,
    pitch_max,
    nacelle_min,
    nacelle_max,
):
    df_out = df_in.dropna(subset=[wind_col, power_col]).copy()

    if df_out.empty:
        return df_out

    df_out = df_out[
        (df_out[wind_col] >= wind_min)
        & (df_out[wind_col] <= wind_max)
        & (df_out[power_col] >= 0)
    ].copy()

    if pitch_col in df_out.columns:
        df_out = df_out[
            df_out[pitch_col].isna()
            | ((df_out[pitch_col] >= pitch_min) & (df_out[pitch_col] <= pitch_max))
        ].copy()

    if nacelle_col in df_out.columns:
        df_out = df_out[
            df_out[nacelle_col].isna()
            | ((df_out[nacelle_col] >= nacelle_min) & (df_out[nacelle_col] <= nacelle_max))
        ].copy()

    return df_out


def build_site_reference_map(ref_raw):
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
            "ReferencePower_kW": ref_power,
        }).dropna()

        ref = ref[
            (ref["ReferenceWindSpeed_mps"] >= 3)
            & (ref["ReferenceWindSpeed_mps"] <= 15)
        ]
        ref = ref.sort_values("ReferenceWindSpeed_mps").drop_duplicates("ReferenceWindSpeed_mps")

        if len(ref) < 2:
            continue

        site_map[site_name] = ref.reset_index(drop=True)

    return site_map


def apply_power_curve_logic(
    df_in,
    ref_curve_raw,
    wind_col,
    power_col,
    status_filter_col,
    wind_std_filter_col,
    pitch_std_filter_col,
):
    required_cols = [status_filter_col, wind_std_filter_col, pitch_std_filter_col]
    missing_deviation_cols = [
        name for name, col in zip(
            ["PCurveStsAve", "WindSpeedStd", "BladePitchStd"],
            required_cols,
        ) if col is None
    ]

    merged_empty = pd.DataFrame(columns=["WindBin", "RefPower", "AvgPower", "Deviation_%"])

    if missing_deviation_cols:
        return {
            "power_curve_df": df_in.iloc[0:0].copy(),
            "merged": merged_empty,
            "avg_dev": None,
            "missing_deviation_cols": missing_deviation_cols,
            "deviation_filter_rows": 0,
        }

    mask_status = df_in[status_filter_col].eq(100)
    mask_wind_std = df_in[wind_std_filter_col].lt(1)
    mask_pitch_std = df_in[pitch_std_filter_col].between(-5, 19, inclusive="both")

    power_curve_df = df_in.loc[mask_status & mask_wind_std & mask_pitch_std].copy()

    ref_min = ref_curve_raw["ReferenceWindSpeed_mps"].min()
    ref_max = ref_curve_raw["ReferenceWindSpeed_mps"].max()

    power_curve_df = power_curve_df[
        power_curve_df[wind_col].between(ref_min, ref_max, inclusive="both")
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
            "missing_deviation_cols": [],
            "deviation_filter_rows": 0,
        }

    power_curve_df["_ReferencePower_kW"] = np.interp(
        power_curve_df[wind_col].to_numpy(dtype=float),
        ref_curve_raw["ReferenceWindSpeed_mps"].to_numpy(dtype=float),
        ref_curve_raw["ReferencePower_kW"].to_numpy(dtype=float),
    )

    power_curve_df["_Deviation_kW"] = (
        power_curve_df[power_col] - power_curve_df["_ReferencePower_kW"]
    )
    power_curve_df["_Shortfall_kW"] = -power_curve_df["_Deviation_kW"]
    power_curve_df["_Deviation_pct"] = np.where(
        power_curve_df["_ReferencePower_kW"] > 0,
        100 * power_curve_df["_Deviation_kW"] / power_curve_df["_ReferencePower_kW"],
        np.nan,
    )
    power_curve_df["_WindBin"] = (
        np.floor(power_curve_df[wind_col] / BIN_SIZE) * BIN_SIZE
    ).round(6)

    actual = power_curve_df.groupby("_WindBin", as_index=False).agg(
        AvgPower=(power_col, "mean"),
        MeanReferencePower=("_ReferencePower_kW", "mean"),
        Deviation_pct=("_Deviation_pct", "mean"),
    ).rename(columns={"_WindBin": "WindBin"})

    ref_plot = ref_curve_raw.copy()
    ref_plot["WindBin"] = ref_plot["ReferenceWindSpeed_mps"]
    ref_plot["RefPower"] = ref_plot["ReferencePower_kW"]
    ref_plot = ref_plot[["WindBin", "RefPower"]]

    merged = ref_plot.merge(actual, on="WindBin", how="left")
    merged["Deviation_%"] = merged["Deviation_pct"]
    merged = merged.drop(columns=["Deviation_pct"], errors="ignore")

    mean_actual = power_curve_df[power_col].mean()
    mean_reference = power_curve_df["_ReferencePower_kW"].mean()

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
        "missing_deviation_cols": [],
        "deviation_filter_rows": len(power_curve_df),
    }


def build_site_analysis(
    all_processed_results,
    ref_curve_raw,
    df_source,
    time_col,
    wind_col,
    power_col,
    status_filter_col,
    wind_std_filter_col,
    pitch_std_filter_col,
    site,
    start_day,
    end_day,
):
    analysis_frames = [
        r["power_curve_df"].copy()
        for r in all_processed_results.values()
        if isinstance(r.get("power_curve_df"), pd.DataFrame) and not r["power_curve_df"].empty
    ]

    if analysis_frames:
        analysis_data = pd.concat(analysis_frames, ignore_index=True)
    else:
        analysis_data = pd.DataFrame()

    filter_count_rows = [["SCADA records in selected date range", int(len(df_source))]]

    if status_filter_col is not None:
        _status_mask = df_source[status_filter_col].eq(100)
        filter_count_rows.append(["PCurveStsAve = 100", int(_status_mask.sum())])
    else:
        _status_mask = pd.Series(False, index=df_source.index)
        filter_count_rows.append(["PCurveStsAve = 100 (column missing)", 0])

    if wind_std_filter_col is not None:
        _wind_std_mask = df_source[wind_std_filter_col].lt(1)
    else:
        _wind_std_mask = pd.Series(False, index=df_source.index)

    if pitch_std_filter_col is not None:
        _pitch_std_mask = df_source[pitch_std_filter_col].between(-5, 19, inclusive="both")
    else:
        _pitch_std_mask = pd.Series(False, index=df_source.index)

    filter_count_rows.extend([
        ["After status and WindSpeedStd < 1", int((_status_mask & _wind_std_mask).sum())],
        ["After all three required filters", int((_status_mask & _wind_std_mask & _pitch_std_mask).sum())],
        ["Compared with selected reference curve", int(len(analysis_data))],
    ])
    filter_counts_df = pd.DataFrame(filter_count_rows, columns=["Filter Stage", "Record Count"])

    if analysis_data.empty:
        return {
            "analysis_data": analysis_data,
            "filter_counts_df": filter_counts_df,
            "overall_df": None,
            "wtg_summary_df": None,
            "fleet_bins_df": None,
            "wtg_bins_df": None,
        }

    analysis_data["WTG"] = analysis_data["Name"].astype(str)
    analysis_data["WindSpeed_mps"] = pd.to_numeric(analysis_data[wind_col], errors="coerce")
    analysis_data["ActualPower_kW"] = pd.to_numeric(analysis_data[power_col], errors="coerce")
    analysis_data["ReferencePower_kW"] = analysis_data["_ReferencePower_kW"]
    analysis_data["Deviation_kW"] = analysis_data["_Deviation_kW"]
    analysis_data["Shortfall_kW"] = analysis_data["_Shortfall_kW"]
    analysis_data["Deviation_pct"] = analysis_data["_Deviation_pct"]
    analysis_data["WindSpeedBin_mps"] = analysis_data["_WindBin"]
    analysis_data["DateTime"] = analysis_data[time_col]

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
        100 * (wtg_summary_df["MeanActualPower_kW"] - wtg_summary_df["MeanReferencePower_kW"])
        / wtg_summary_df["MeanReferencePower_kW"],
        np.nan,
    )
    wtg_summary_df = wtg_summary_df.sort_values("AggregateDeviation_pct", na_position="last")

    ref_min = ref_curve_raw["ReferenceWindSpeed_mps"].min()
    ref_max = ref_curve_raw["ReferenceWindSpeed_mps"].max()

    overall_df = pd.DataFrame([
        ["Selected site", site],
        ["Selected date range", f"{start_day} to {end_day}"],
        ["Turbines compared", int(analysis_data["WTG"].nunique())],
        ["Records compared to reference curve", int(len(analysis_data))],
        ["Reference curve min wind speed (m/s)", ref_min],
        ["Reference curve max wind speed (m/s)", ref_max],
        ["Filter 1", "PCurveStsAve = 100"],
        ["Filter 2", "WindSpeedStd < 1"],
        ["Filter 3", "-5 <= BladePitchStd <= 19"],
        ["Wind-speed bin width (m/s)", BIN_SIZE],
        ["Deviation sign", "Actual power - reference power; negative means below reference"],
        ["Aggregate turbine deviation", "100 * (mean actual power - mean reference power) / mean reference power"],
    ], columns=["Metric", "Value"])

    return {
        "analysis_data": analysis_data,
        "filter_counts_df": filter_counts_df,
        "overall_df": overall_df,
        "wtg_summary_df": wtg_summary_df,
        "fleet_bins_df": fleet_bins_df,
        "wtg_bins_df": wtg_bins_df,
    }


# =========================
# UI HEADER
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

st.title("Power Curve Analytics Dashboard")

# =========================
# FILE UPLOADS
# =========================
st.sidebar.subheader("Upload Files")
scada_file = st.sidebar.file_uploader("Upload 1-min / 10-min SCADA CSV", type=["csv"], key="scada_file")
ref_file = st.sidebar.file_uploader("Upload Reference Excel", type=["xlsx"], key="ref_file")

if scada_file is None:
    st.warning("Please upload SCADA/static CSV file.")
    st.stop()

if ref_file is None:
    st.warning("Please upload reference Excel file.")
    st.stop()

# =========================
# LOAD SCADA
# =========================
@st.cache_data(show_spinner=True)
def load_scada(file):
    chunks = pd.read_csv(file, chunksize=200000, low_memory=False, engine="c")
    df_local = pd.concat(chunks, ignore_index=True)
    df_local.columns = df_local.columns.str.strip()
    return df_local


with st.spinner("Loading SCADA file..."):
    df = load_scada(scada_file)

if "Name" not in df.columns:
    st.error("SCADA file must contain 'Name' column.")
    st.stop()

available_columns = list(df.columns)

auto_time = detect_column(available_columns, ["Time", "Timestamp", "DateTime", "LocalTime"])
auto_wind = detect_column(available_columns, ["WindSpeedAve", "Wind Speed Ave", "WindSpeed", "Wind"])
auto_power = detect_column(
    available_columns,
    [
        "ActivePWAve",
        "ActivePowerAve",
        "PowerAve",
        "ActivePower",
        "Power",
        "GrdProdPwrAve",
        "GrdProdPwrAct",
        "WTGActivePowerAve",
        "OutputPowerAve",
    ],
)
auto_pitch = detect_column(available_columns, ["BladePitchAve", "PitchAve", "Pitch"])
auto_nacelle = detect_column(available_columns, ["NacellePositionAve", "NacelleAve", "Nacelle Position"])

status_filter_col = detect_column(available_columns, ["PCurveStsAve"])
wind_std_filter_col = detect_column(available_columns, ["WindSpeedStd"])
pitch_std_filter_col = detect_column(available_columns, ["BladePitchStd"])

st.sidebar.markdown("### Column Mapping")

time_col = st.sidebar.selectbox(
    "Time Column",
    available_columns,
    index=available_columns.index(auto_time) if auto_time in available_columns else 0,
)
wind_col = st.sidebar.selectbox(
    "Wind Column",
    available_columns,
    index=available_columns.index(auto_wind) if auto_wind in available_columns else 0,
)
power_col = st.sidebar.selectbox(
    "Power Column",
    available_columns,
    index=available_columns.index(auto_power) if auto_power in available_columns else 0,
)
pitch_col = st.sidebar.selectbox(
    "Pitch Column",
    available_columns,
    index=available_columns.index(auto_pitch) if auto_pitch in available_columns else 0,
)
nacelle_col = st.sidebar.selectbox(
    "Nacelle Column",
    available_columns,
    index=available_columns.index(auto_nacelle) if auto_nacelle in available_columns else 0,
)

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
df[pitch_col] = pd.to_numeric(df[pitch_col], errors="coerce")
df[nacelle_col] = pd.to_numeric(df[nacelle_col], errors="coerce")

for _filter_col in [status_filter_col, wind_std_filter_col, pitch_std_filter_col]:
    if _filter_col is not None:
        df[_filter_col] = pd.to_numeric(df[_filter_col], errors="coerce")

df["Name"] = df["Name"].astype(str).str.strip()
df = df.dropna(subset=["Name"])

if df.empty:
    st.warning("No valid SCADA rows after parsing.")
    st.stop()

# =========================
# DATE FILTER
# =========================
st.sidebar.markdown("### Date Range")
max_ts = df[time_col].max()
base_date = max_ts.normalize().date() if pd.notna(max_ts) else pd.Timestamp.today().date()

DEFAULT_START = base_date - timedelta(days=15)
DEFAULT_END = base_date

if "manual_start_date" not in st.session_state:
    st.session_state.manual_start_date = DEFAULT_START
if "manual_end_date" not in st.session_state:
    st.session_state.manual_end_date = DEFAULT_END

date_option = st.sidebar.selectbox(
    "Date Option",
    ["Clear", "Today", "This Week", "This Month", "Last Week", "Last Month", "Manual (Calendar)"],
    key="date_option",
)

if date_option == "Manual (Calendar)":
    start_day = st.sidebar.date_input("Start Date", value=st.session_state.manual_start_date, key="manual_start_date")
    end_day = st.sidebar.date_input("End Date", value=st.session_state.manual_end_date, key="manual_end_date")
elif date_option == "Clear":
    st.session_state.manual_start_date = DEFAULT_START
    st.session_state.manual_end_date = DEFAULT_END
    start_day = DEFAULT_START
    end_day = DEFAULT_END
else:
    rng = compute_preset_range(date_option, pd.Timestamp.today())
    start_day, end_day = rng if rng else (DEFAULT_START, DEFAULT_END)

df["_date_only"] = df[time_col].dt.date
df = df[(df["_date_only"] >= start_day) & (df["_date_only"] <= end_day)].drop(columns=["_date_only"])

if df.empty:
    st.warning("No SCADA data available for selected date range.")
    st.stop()

# =========================
# TURBINES
# =========================
all_turbines = sorted(df["Name"].dropna().unique())

mode = st.sidebar.radio(
    "Select View",
    ["Single Turbine", "Compare Turbines", "Show All Turbines"],
    key="mode_radio",
)

if mode == "Single Turbine":
    turbines_to_show = [st.sidebar.selectbox("Select Turbine", all_turbines, key="single_turbine")]
elif mode == "Compare Turbines":
    turbines_to_show = st.sidebar.multiselect("Select Turbines", all_turbines, default=all_turbines, key="compare_turbines")
else:
    turbines_to_show = all_turbines

if not turbines_to_show:
    st.warning("Please select at least one turbine.")
    st.stop()

# =========================
# LOAD REFERENCE
# =========================
@st.cache_data
def load_reference(reference_file):
    ref_raw = pd.read_excel(reference_file, sheet_name="Power Curve", header=None)
    site_map = build_site_reference_map(ref_raw)
    site_names = list(site_map.keys())
    if not site_names:
        raise ValueError("No valid site reference curves found in Power Curve sheet.")
    return site_names, site_map


try:
    ref_site_names, ref_site_map = load_reference(ref_file)
except Exception as e:
    st.error("Unable to read reference Excel / Power Curve sheet.")
    st.code(str(e))
    st.stop()

site = st.sidebar.selectbox("Select Site / Reference Column", ref_site_names)
ref_curve_raw = ref_site_map[site].copy()

st.subheader(f"Reference Selected: {site}")
st.markdown(f"Date Range: {start_day} → {end_day}")

# =========================
# FILTERS
# =========================
st.sidebar.markdown("### Filters")
pitch_min = st.sidebar.number_input("Pitch Min", value=-5.0)
pitch_max = st.sidebar.number_input("Pitch Max", value=5.0)
nacelle_min = st.sidebar.number_input("Nacelle Min", value=0.0)
nacelle_max = st.sidebar.number_input("Nacelle Max", value=360.0)
wind_min = st.sidebar.number_input("Wind Min", value=3.0)
wind_max = st.sidebar.number_input("Wind Max", value=15.0)

# =========================
# FORMULA / LOGIC DISPLAY
# =========================
with st.expander("Applied Formula and Filtering Logic", expanded=False):
    st.markdown("### Selected Site Logic")
    st.markdown(f"- The power-curve calculation uses the selected site's reference column: **{site}**.")
    st.markdown("- This matches the standalone analysis logic, but dynamically for any site chosen in the Excel file.")

    st.markdown("### Required Power-Curve Filters")
    st.code(
        "\n".join(
            [
                f"{status_filter_col} == 100" if status_filter_col else "PCurveStsAve column missing",
                f"{wind_std_filter_col} < 1" if wind_std_filter_col else "WindSpeedStd column missing",
                f"-5 <= {pitch_std_filter_col} <= 19" if pitch_std_filter_col else "BladePitchStd column missing",
            ]
        ),
        language="python",
    )

    st.markdown("### Dashboard Filters")
    st.code(
        "\n".join(
            [
                f"{wind_col} is not null",
                f"{power_col} is not null",
                f"{wind_min} <= {wind_col} <= {wind_max}",
                f"{power_col} >= 0",
                f"{pitch_col} is null OR ({pitch_min} <= {pitch_col} <= {pitch_max})",
                f"{nacelle_col} is null OR ({nacelle_min} <= {nacelle_col} <= {nacelle_max})",
            ]
        ),
        language="python",
    )

    st.markdown("### Power Curve Formula")
    st.code(
        "\n".join(
            [
                "ReferencePower = linear interpolation from selected site curve",
                "No extrapolation outside reference wind-speed range",
                f"WindSpeedBin = floor({wind_col} / {BIN_SIZE}) * {BIN_SIZE}",
                "Deviation_kW = ActualPower - ReferencePower",
                "Shortfall_kW = -Deviation_kW",
                "Deviation_pct = 100 * Deviation_kW / ReferencePower",
                "AggregateDeviation_pct = 100 * (mean ActualPower - mean ReferencePower) / mean ReferencePower",
            ]
        ),
        language="python",
    )

# =========================
# PROCESS
# =========================
def process_turbine(t):
    df_t_all = df[df["Name"] == t].copy()
    raw_rows = len(df_t_all)

    result = {
        "turbine": t,
        "raw_rows": raw_rows,
        "status": "No Data",
        "comment": "No data available",
        "df_dashboard": pd.DataFrame(columns=[wind_col, power_col, nacelle_col]),
        "df_nacelle": pd.DataFrame(columns=[wind_col, nacelle_col]),
        "power_curve_df": pd.DataFrame(columns=[wind_col, power_col]),
        "merged": pd.DataFrame(columns=["WindBin", "RefPower", "AvgPower", "Deviation_%"]),
        "avg_dev": None,
        "avg_nacelle": None,
        "nacelle_curve": pd.DataFrame(),
        "dashboard_power_sample_count": 0,
        "deviation_filter_rows": 0,
        "nacelle_sample_count": 0,
        "missing_deviation_cols": [],
    }

    if raw_rows == 0:
        return result

    df_dashboard = build_dashboard_filtered_df(
        df_t_all,
        wind_col,
        power_col,
        pitch_col,
        nacelle_col,
        wind_min,
        wind_max,
        pitch_min,
        pitch_max,
        nacelle_min,
        nacelle_max,
    )

    if df_dashboard.empty:
        result["status"] = "Filtered Out"
        result["comment"] = "All rows removed by dashboard filters"
        return result

    result["df_dashboard"] = df_dashboard

    dashboard_power_samples_df = df_dashboard.dropna(subset=[wind_col, power_col]).copy()
    nacelle_samples_df = df_dashboard.dropna(subset=[wind_col, nacelle_col]).copy()

    result["dashboard_power_sample_count"] = len(dashboard_power_samples_df)
    result["nacelle_sample_count"] = len(nacelle_samples_df)
    result["df_nacelle"] = nacelle_samples_df

    if nacelle_samples_df.empty:
        nacelle_comment = "Nacelle average not available"
    else:
        result["avg_nacelle"] = nacelle_samples_df[nacelle_col].mean()
        result["nacelle_curve"] = build_metric_curve(
            nacelle_samples_df,
            wind_col,
            nacelle_col,
        )
        nacelle_comment = generate_nacelle_comment(result["avg_nacelle"])

    pc_result = apply_power_curve_logic(
        df_dashboard,
        ref_curve_raw,
        wind_col,
        power_col,
        status_filter_col,
        wind_std_filter_col,
        pitch_std_filter_col,
    )

    result["power_curve_df"] = pc_result["power_curve_df"]
    result["merged"] = pc_result["merged"]
    result["avg_dev"] = pc_result["avg_dev"]
    result["missing_deviation_cols"] = pc_result["missing_deviation_cols"]
    result["deviation_filter_rows"] = pc_result["deviation_filter_rows"]

    if result["avg_dev"] is None:
        result["status"] = "Low Data"
        if result["missing_deviation_cols"]:
            result["comment"] = (
                "Deviation unavailable: missing required SCADA columns "
                + ", ".join(result["missing_deviation_cols"])
                + f" | {nacelle_comment}"
            )
        elif result["power_curve_df"].empty:
            result["comment"] = (
                "No rows passed required power-curve filters "
                "(PCurveStsAve=100, WindSpeedStd<1, -5<=BladePitchStd<=19)"
                + f" | {nacelle_comment}"
            )
        else:
            result["comment"] = f"Insufficient overlap with reference | {nacelle_comment}"
    else:
        result["status"] = "OK"
        result["comment"] = f"{generate_comment(result['avg_dev'])} | {nacelle_comment}"

    return result

# =========================
# PLOT
# =========================
def plot_power_curve(power_curve_df, merged, title, dev, comment, deviation_sample_count):
    fig = go.Figure()

    if merged is not None and not merged.empty:
        fig.add_trace(
            go.Scatter(
                x=merged["WindBin"],
                y=merged["RefPower"],
                mode="lines",
                line=dict(color="#d62728", width=2.5, dash="dash"),
                name="Reference Power Curve",
                hovertemplate="Wind Speed: %{x:.2f} m/s<br>Reference Power: %{y:.2f} kW<extra></extra>",
            )
        )

    if not power_curve_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=power_curve_df[wind_col],
                y=power_curve_df[power_col],
                mode="markers",
                marker=dict(
                    size=5,
                    color="rgba(99, 102, 241, 0.18)",
                    line=dict(width=0),
                ),
                name=f"Filtered 10-min SCADA ({deviation_sample_count})",
                hovertemplate=(
                    f"{wind_col}: " + "%{x:.2f} m/s<br>"
                    f"{power_col}: " + "%{y:.2f} kW<extra>Filtered SCADA</extra>"
                ),
            )
        )

    avg_curve = merged.dropna(subset=["AvgPower"]).copy() if merged is not None and not merged.empty else pd.DataFrame()
    if not avg_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=avg_curve["WindBin"],
                y=avg_curve["AvgPower"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=7, color="#1f77b4"),
                name="SCADA mean by 0.5 m/s bin",
                hovertemplate="Wind Speed Bin: %{x:.2f} m/s<br>Average Power: %{y:.2f} kW<extra></extra>",
            )
        )

    y_max = 3500
    if merged is not None and not merged.empty and merged["RefPower"].notna().any():
        y_max = max(y_max, float(np.nanmax(merged["RefPower"])) + 150)
    if merged is not None and not merged.empty and merged["AvgPower"].notna().any():
        y_max = max(y_max, float(np.nanmax(merged["AvgPower"])) + 150)
    if not power_curve_df.empty and power_curve_df[power_col].notna().any():
        y_max = max(y_max, float(np.nanmax(power_curve_df[power_col])) + 150)

    fig.update_layout(
        title=f"{title} - Power Curve Analysis",
        xaxis=dict(
            title="Wind Speed (m/s)",
            range=[3, 15],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Active Power (kW)",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
        ),
        height=520,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=60, r=40, t=70, b=60),
    )

    return fig


def plot_nacelle_graph(df_t, nacelle_curve, title, avg_nacelle, nacelle_sample_count):
    nac_txt = "NA" if avg_nacelle is None or pd.isna(avg_nacelle) else round(avg_nacelle, 2)

    fig = go.Figure()

    if not df_t.empty:
        fig.add_trace(
            go.Scattergl(
                x=df_t[wind_col],
                y=df_t[nacelle_col],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.30,
                    color="rgba(0,128,128,0.55)",
                    line=dict(width=0),
                ),
                name=f"Nacelle Samples ({nacelle_sample_count})",
                hovertemplate=(
                    f"{wind_col}: " + "%{x:.2f} m/s<br>"
                    f"{nacelle_col}: " + "%{y:.2f}°<extra>Nacelle Sample</extra>"
                ),
            )
        )

    if not nacelle_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=nacelle_curve["WindBin"],
                y=nacelle_curve["AvgMetric"],
                mode="lines+markers",
                line=dict(width=2.5, color="#008080"),
                marker=dict(size=6, color="#008080"),
                name="Nacelle mean by 0.5 m/s bin",
                hovertemplate=(
                    "Wind Speed Bin: %{x:.2f} m/s<br>"
                    "Avg Nacelle: %{y:.2f}°<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{title} - Nacelle Graph",
        xaxis=dict(
            title="Wind Speed (m/s)",
            range=[3, 15],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Nacelle Position (°)",
            range=[0, 360],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.12)",
            zeroline=False,
        ),
        height=480,
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.85)"
        ),
        margin=dict(l=60, r=40, t=70, b=60),
    )
    return fig

# =========================
# DISPLAY
# =========================
results = []
figures = []
processed_results = {}

for t in turbines_to_show:
    res = process_turbine(t)
    processed_results[t] = res

    st.markdown(f"## Turbine: {t}")
    tab1, tab2 = st.tabs(["Power Curve", "Nacelle Graph"])

    with tab1:
        fig_power = plot_power_curve(
            res["power_curve_df"],
            res["merged"],
            res["turbine"],
            res["avg_dev"],
            res["comment"],
            res["deviation_filter_rows"],
        )
        st.plotly_chart(fig_power, use_container_width=True)
        st.caption(f"Deviation: {'NA' if res['avg_dev'] is None else round(res['avg_dev'], 2)}%")
        st.caption(f"Filtered Power Samples: {res['deviation_filter_rows']}")
        st.caption(res["comment"])

    with tab2:
        fig_nacelle = plot_nacelle_graph(
            res["df_nacelle"],
            res["nacelle_curve"],
            res["turbine"],
            res["avg_nacelle"],
            res["nacelle_sample_count"],
        )
        st.plotly_chart(fig_nacelle, use_container_width=True)
        st.caption(f"Nacelle Sample Formula: X = {wind_col}, Y = {nacelle_col}")
        st.caption(f"Nacelle Samples Count: {res['nacelle_sample_count']}")

    st.divider()

    figures.append((t, fig_power, res["comment"]))

    results.append(
        {
            "Turbine": t,
            "Deviation_%": None if res["avg_dev"] is None else round(res["avg_dev"], 2),
            "Avg_Nacelle": None if res["avg_nacelle"] is None else round(res["avg_nacelle"], 2),
            "Nacelle_Band": get_nacelle_band(res["avg_nacelle"]),
            "Dashboard_Power_Samples": res["dashboard_power_sample_count"],
            "Deviation_Filtered_Samples": res["deviation_filter_rows"],
            "Nacelle_Samples": res["nacelle_sample_count"],
            "Status": res["status"],
            "Rows": res["raw_rows"],
            "Comment": res["comment"],
        }
    )

# =========================
# RANKING
# =========================
st.subheader("Turbine Ranking")
results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values(by="Deviation_%", na_position="last")
    st.dataframe(results_df, use_container_width=True)

# =========================
# DETAILED SITE ANALYSIS
# =========================
st.subheader(f"Detailed Power-Curve Analysis — {site}")
st.caption(
    "The summary tables below use the same logic as the standalone script: "
    "required filters, selected-site reference interpolation, no extrapolation, "
    "and 0.5 m/s wind-speed bins."
)

for _t in all_turbines:
    if _t not in processed_results:
        processed_results[_t] = process_turbine(_t)

site_analysis = build_site_analysis(
    processed_results,
    ref_curve_raw,
    df,
    time_col,
    wind_col,
    power_col,
    status_filter_col,
    wind_std_filter_col,
    pitch_std_filter_col,
    site,
    start_day,
    end_day,
)

analysis_data = site_analysis["analysis_data"]
filter_counts_df = site_analysis["filter_counts_df"]
overall_df = site_analysis["overall_df"]
wtg_summary_df = site_analysis["wtg_summary_df"]
fleet_bins_df = site_analysis["fleet_bins_df"]
wtg_bins_df = site_analysis["wtg_bins_df"]

if not analysis_data.empty:
    fleet_fig = go.Figure()
    fleet_fig.add_trace(go.Scatter(
        x=ref_curve_raw["ReferenceWindSpeed_mps"],
        y=ref_curve_raw["ReferencePower_kW"],
        mode="lines",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        name="Reference Power Curve",
    ))
    fleet_fig.add_trace(go.Scattergl(
        x=analysis_data["WindSpeed_mps"],
        y=analysis_data["ActualPower_kW"],
        mode="markers",
        marker=dict(size=5, color="rgba(99, 102, 241, 0.18)", line=dict(width=0)),
        name="Filtered 10-min SCADA",
    ))
    fleet_fig.add_trace(go.Scatter(
        x=fleet_bins_df["MeanWindSpeed_mps"],
        y=fleet_bins_df["MeanActualPower_kW"],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=7, color="#1f77b4"),
        name="SCADA mean by 0.5 m/s bin",
    ))
    fleet_fig.update_layout(
        title=f"{site} - Power Curve Analysis",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=520,
        template="plotly_white"
    )
    st.plotly_chart(fleet_fig, use_container_width=True)

    deviation_fig = go.Figure()
    deviation_fig.add_hline(y=0, line_width=1, line_color="gray")
    deviation_fig.add_trace(go.Scatter(
        x=fleet_bins_df["MeanWindSpeed_mps"],
        y=fleet_bins_df["MeanDeviation_kW"],
        mode="lines+markers",
        line=dict(width=2),
        name="Mean deviation",
    ))
    deviation_fig.update_layout(
        title=f"{site} - Power Deviation by Wind-Speed Bin",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Mean Deviation (Actual - Reference, kW)",
        height=420,
        template="plotly_white",
    )
    st.plotly_chart(deviation_fig, use_container_width=True)

    tab_overall, tab_filters, tab_wtg, tab_fleet_bins, tab_wtg_bins, tab_details = st.tabs([
        "Overall Summary", "Filter Counts", "WTG Summary", "Fleet Wind Bins", "WTG Wind Bins", "Filtered SCADA Details"
    ])

    with tab_overall:
        st.dataframe(overall_df, use_container_width=True, hide_index=True)
    with tab_filters:
        st.dataframe(filter_counts_df, use_container_width=True, hide_index=True)
    with tab_wtg:
        st.dataframe(wtg_summary_df, use_container_width=True, hide_index=True)
    with tab_fleet_bins:
        st.dataframe(fleet_bins_df, use_container_width=True, hide_index=True)
    with tab_wtg_bins:
        st.dataframe(wtg_bins_df, use_container_width=True, hide_index=True)
    with tab_details:
        detail_cols = [
            "WTG", "DateTime", status_filter_col, wind_std_filter_col, pitch_std_filter_col,
            "WindSpeed_mps", "ActualPower_kW", "ReferencePower_kW", "Deviation_kW",
            "Shortfall_kW", "Deviation_pct", "WindSpeedBin_mps"
        ]
        detail_cols = [c for c in detail_cols if c is not None and c in analysis_data.columns]
        st.dataframe(analysis_data[detail_cols], use_container_width=True, hide_index=True)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall Summary", index=False)
        filter_counts_df.to_excel(writer, sheet_name="Filter Counts", index=False)
        wtg_summary_df.to_excel(writer, sheet_name="WTG Summary", index=False)
        fleet_bins_df.to_excel(writer, sheet_name="Fleet Wind Bins", index=False)
        wtg_bins_df.to_excel(writer, sheet_name="WTG Wind Bins", index=False)

        detail_cols = [
            "WTG", "DateTime", status_filter_col, wind_std_filter_col, pitch_std_filter_col,
            "WindSpeed_mps", "ActualPower_kW", "ReferencePower_kW", "Deviation_kW",
            "Shortfall_kW", "Deviation_pct", "WindSpeedBin_mps"
        ]
        detail_cols = [c for c in detail_cols if c is not None and c in analysis_data.columns]
        analysis_data[detail_cols].to_excel(writer, sheet_name="Filtered SCADA Details", index=False)

        ref_curve_raw.to_excel(writer, sheet_name="Reference Curve", index=False)

    excel_buffer.seek(0)
    st.download_button(
        "Download Complete Power-Curve Analysis (Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"{str(site).replace(' ', '_')}_Power_Curve_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("No SCADA samples passed the required filters and selected reference-curve range for this site/date selection.")
    st.dataframe(filter_counts_df, use_container_width=True, hide_index=True)

# =========================
# PDF DOWNLOAD
# =========================
try:
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=landscape(A4))
    width, height = landscape(A4)

    if os.path.exists(logo_path):
        pdf.drawImage(logo_path, 30, height - 80, width=120, height=40)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(170, height - 40, "Power Curve Analytics Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(170, height - 60, f"Site: {site}")
    pdf.drawString(170, height - 75, f"Date Range: {start_day} to {end_day}")

    y = height - 120

    for turbine, fig, comment in figures:
        if KALEIDO_AVAILABLE:
            try:
                img = fig.to_image(format="png")
                img_reader = ImageReader(io.BytesIO(img))

                if y < 260:
                    pdf.showPage()
                    y = height - 60

                pdf.drawImage(img_reader, 30, y - 220, width=360, height=200)
                pdf.setFont("Helvetica-Bold", 11)
                pdf.drawString(420, y - 40, turbine)
                pdf.setFont("Helvetica", 10)
                pdf.drawString(420, y - 60, comment[:110])
                y -= 240
            except Exception:
                pass

    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(30, height - 40, "Turbine Ranking Summary")
    y = height - 80
    pdf.setFont("Helvetica", 10)

    if not results_df.empty:
        for _, row in results_df.iterrows():
            dev_text = "NA" if pd.isna(row["Deviation_%"]) else row["Deviation_%"]
            nac_text = "NA" if pd.isna(row["Avg_Nacelle"]) else row["Avg_Nacelle"]
            line = (
                f"{row['Turbine']} | Dev: {dev_text}% | "
                f"Nacelle: {nac_text}° | Band: {row['Nacelle_Band']} | "
                f"Dashboard Power Samples: {row['Dashboard_Power_Samples']} | "
                f"Deviation Samples: {row['Deviation_Filtered_Samples']} | "
                f"Nacelle Samples: {row['Nacelle_Samples']} | "
                f"{row['Status']}"
            )
            pdf.drawString(40, y, line[:170])
            y -= 20
            if y < 40:
                pdf.showPage()
                y = height - 40

    pdf.save()
    pdf_buffer.seek(0)

    st.download_button(
        label="Download Full Dashboard Report (PDF)",
        data=pdf_buffer.getvalue(),
        file_name="WindFarm_Full_Report.pdf",
        mime="application/pdf",
    )

except Exception as e:
    st.error("PDF generation failed")
    st.code(str(e))
