import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
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

def safe_savgol(series, window=7, poly=2):
    s = series.copy()
    valid = s.notna()

    valid_count = valid.sum()
    if valid_count < 5:
        return s

    if window > valid_count:
        window = valid_count if valid_count % 2 == 1 else valid_count - 1

    if window < 3:
        return s

    if poly >= window:
        poly = max(1, window - 1)

    try:
        s.loc[valid] = savgol_filter(s.loc[valid], window, poly)
    except Exception:
        pass
    return s

def build_metric_curve(df_t, wind_col, metric_col, smooth=False, smooth_window=7, smooth_poly=2):
    tmp = df_t[[wind_col, metric_col]].dropna().copy()

    if tmp.empty:
        return pd.DataFrame(columns=["WindBin", "AvgMetric"])

    tmp["WindBin"] = (np.floor(tmp[wind_col] / BIN_SIZE) * BIN_SIZE).round(6)
    out = tmp.groupby("WindBin", as_index=False).agg(AvgMetric=(metric_col, "mean"))

    if smooth and not out.empty:
        out["AvgMetric"] = safe_savgol(out["AvgMetric"], window=smooth_window, poly=smooth_poly)

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
auto_power = detect_column(available_columns, [
    "ActivePWAve", "ActivePowerAve", "PowerAve", "ActivePower", "Power",
    "GrdProdPwrAve", "GrdProdPwrAct", "WTGActivePowerAve", "OutputPowerAve"
])
auto_pitch = detect_column(available_columns, ["BladePitchAve", "PitchAve", "Pitch"])
auto_nacelle = detect_column(available_columns, ["NacellePositionAve", "NacelleAve", "Nacelle Position"])

st.sidebar.markdown("### Column Mapping")

time_col = st.sidebar.selectbox(
    "Time Column",
    available_columns,
    index=available_columns.index(auto_time) if auto_time in available_columns else 0
)
wind_col = st.sidebar.selectbox(
    "Wind Column",
    available_columns,
    index=available_columns.index(auto_wind) if auto_wind in available_columns else 0
)
power_col = st.sidebar.selectbox(
    "Power Column",
    available_columns,
    index=available_columns.index(auto_power) if auto_power in available_columns else 0
)
pitch_col = st.sidebar.selectbox(
    "Pitch Column",
    available_columns,
    index=available_columns.index(auto_pitch) if auto_pitch in available_columns else 0
)
nacelle_col = st.sidebar.selectbox(
    "Nacelle Column",
    available_columns,
    index=available_columns.index(auto_nacelle) if auto_nacelle in available_columns else 0
)

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
df[pitch_col] = pd.to_numeric(df[pitch_col], errors="coerce")
df[nacelle_col] = pd.to_numeric(df[nacelle_col], errors="coerce")
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
    key="date_option"
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
    key="mode_radio"
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
    site_headers = ref_raw.iloc[2].copy()
    site_names = [str(x) for x in site_headers[1:].tolist()]
    return ref_raw, site_names

try:
    ref_raw, ref_site_names = load_reference(ref_file)
except Exception as e:
    st.error("Unable to read reference Excel / Power Curve sheet.")
    st.code(str(e))
    st.stop()

site = st.sidebar.selectbox("Select Site / Reference Column", ref_site_names)

def load_reference_curve(ref_raw, selected_site):
    site_headers = ref_raw.iloc[2].copy()
    wind_series = pd.to_numeric(ref_raw.iloc[3:, 0], errors="coerce")

    matched_col = None
    for col_idx in range(1, ref_raw.shape[1]):
        header_text_raw = site_headers.iloc[col_idx]
        if str(header_text_raw).strip() == selected_site:
            matched_col = col_idx
            break

    if matched_col is None:
        st.error("Selected site not found in reference sheet.")
        st.stop()

    ref_power = pd.to_numeric(ref_raw.iloc[3:, matched_col], errors="coerce")

    ref = pd.DataFrame({
        "WindSpeed": wind_series,
        "RefPower": ref_power
    }).dropna()

    ref = ref[(ref["WindSpeed"] >= 3) & (ref["WindSpeed"] <= 15)]

    if ref.empty:
        st.error("No valid reference data found.")
        st.stop()

    wind_bins = np.arange(3, 15.5, BIN_SIZE)
    ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

    return pd.DataFrame({"WindBin": wind_bins, "RefPower": ref_interp})

ref_curve = load_reference_curve(ref_raw, site)

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
# FORMULA DISPLAY
# =========================
with st.expander("Applied Formula and Filtering Logic", expanded=False):
    st.markdown(f"""
### Scatter Logic

**Power Sample Scatter**
- **X = {wind_col}**
- **Y = {power_col}**

**Nacelle Sample Scatter**
- **X = {wind_col}**
- **Y = {nacelle_col}**

---

### Filtering Logic

A row is kept only when:

**Base condition for power curve**
```python
{wind_col} is not null
{power_col} is not null
{wind_min} <= {wind_col} <= {wind_max}
{power_col} >= 0
