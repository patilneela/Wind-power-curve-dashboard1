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

try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__)
REF_FILE_PATH = os.path.join(BASE_DIR, "reference.xlsx")
SITE_MASTER_XLSX = os.path.join(BASE_DIR, "site_master.xlsx")
SITE_MASTER_CSV = os.path.join(BASE_DIR, "site_master.csv")
logo_path = os.path.join(BASE_DIR, "Envision.png")
BIN_SIZE = 0.5

def get_site_master_path():
    if os.path.exists(SITE_MASTER_XLSX):
        return SITE_MASTER_XLSX
    if os.path.exists(SITE_MASTER_CSV):
        return SITE_MASTER_CSV
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

def get_nacelle_band(avg_nacelle):
    if avg_nacelle is None or pd.isna(avg_nacelle):
        return "NA"
    if 0 <= avg_nacelle < 60:
        return "0-60"
    elif 60 <= avg_nacelle < 120:
        return "60-120"
    elif 120 <= avg_nacelle < 180:
        return "120-180"
    elif 180 <= avg_nacelle <= 270:
        return "180-270"
    return "Out of Range"

def generate_nacelle_comment(avg_nacelle):
    if avg_nacelle is None or pd.isna(avg_nacelle):
        return "Nacelle average not available"
    band = get_nacelle_band(avg_nacelle)
    val = round(avg_nacelle, 2)
    if band == "Out of Range":
        return f"NacelleAvg: {val}° → Out of configured range"
    return f"NacelleAvg: {val}° → Band {band}"

def generate_performance_comment(dev):
    if dev is None or pd.isna(dev):
        return "Deviation not available"
    dev = round(dev, 2)
    if dev < -72:
        return f"Dev: {dev}% → Extreme issue (Data unreliable)"
    elif dev < -10:
        return f"Dev: {dev}% → Severe underperformance (Blade/Dust/Yaw issue)"
    elif dev < -2:
        return f"Dev: {dev}% → Underperformance (Control/availability)"
    elif dev > 72:
        return f"Dev: {dev}% → Abnormal high (Sensor/Data issue)"
    elif dev > 8:
        return f"Dev: {dev}% → High overperformance"
    elif dev > 2:
        return f"Dev: {dev}% → Slight overperformance"
    return f"Dev: {dev}% → Normal performance"

def generate_pcurve_status_comment(avg_status):
    if avg_status is None or pd.isna(avg_status):
        return "PCurve status not available"
    return f"PCurveStsAve: {round(avg_status, 2)}"

def safe_savgol(series, window=7, poly=2):
    s = series.copy()
    valid = s.notna()
    if valid.sum() < window:
        return s
    try:
        s.loc[valid] = savgol_filter(s.loc[valid], window, poly)
    except Exception:
        pass
    return s

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

st.title("Power Curve Analytics Report")

DEFAULT_SITE_CAPACITY = {
    site: 3.3 for site in [
        "CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2",
        "AMP_Energy","Wanki","CleanMax Motadevaliya","Ayana Amerli","Mahadev PH1",
        "Blupine-I, Ambada-GJ","ACME Shapar","FP_Kudligi","Sprng TN","Otha Pithalpur-GJ",
        "AMGEPL,Kurnool AP","ReNew1_Gadag","partner Ottapidaum","Cleanmax SANATHALI",
        "Cleanmax Babra","RenfraEnergy Trichy","RENEW-03 Sholapur","Renew2 Chandwad",
        "ReNew-4 Patoda","Clean max Jagalur","Sembcorp Tuticorin","Renew-4 Kudligi",
        "Renew Otha","Cleanmax Honavad","Blueleaf Agar","JSW_Sandur","India_Hero_Doni",
    ]
}

@st.cache_data
def load_site_capacity():
    capacity = dict(DEFAULT_SITE_CAPACITY)
    path = get_site_master_path()
    if path is None:
        return capacity
    try:
        sm = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
        sm.columns = [c.strip() for c in sm.columns]
        site_col = None
        cap_col = None
        for c in sm.columns:
            if c.lower() in ["site", "site_name", "sitename", "plant", "project"]:
                site_col = c
            if c.lower() in ["capacity_mw", "capacity", "turbine_capacity_mw", "mw"]:
                cap_col = c
        if site_col is None or cap_col is None:
            st.warning("site_master file found but columns not recognized. Required: Site + Capacity_MW.")
            return capacity
        sm = sm[[site_col, cap_col]].dropna()
        sm[site_col] = sm[site_col].astype(str).str.strip()
        sm[cap_col] = pd.to_numeric(sm[cap_col], errors="coerce")
        sm = sm.dropna()
        for _, row in sm.iterrows():
            capacity[row[site_col]] = float(row[cap_col])
        return capacity
    except Exception as e:
        st.warning("Failed to read site_master. Using default site list.")
        st.code(str(e))
        return capacity

SITE_CAPACITY = load_site_capacity()

SITE_REFERENCE_ALIASES = {
    "CleanMax Motadevaliya": ["cleanmaxgujarat"],
    "Renew-4 Kudligi": ["renew4kudligika"],
    "Renew Otha": ["renew4othagj", "renew4otha", "otha"],
    "Otha Pithalpur-GJ": ["renewpithalurgj", "pithalur", "pithalpur"],
    "Clean max Jagalur": ["cleanmaxjagalurka", "jagalur"],
    "partner Ottapidaum": ["fourthpartnerottapidaramtn", "ottapidaram"],
    "Sembcorp Tuticorin": ["sembcorptuticorintn", "tuticorin"],
    "AMGEPL,Kurnool AP": ["amgpelkurnool", "kurnool"],
    "JSW_Sandur": ["jswsandurka", "jswsandur"],
    "Ayana Amerli": ["ayanaamreligj", "ayanaamreli", "amreli"],
    "Sprng TN": ["sprngmulanurtn", "mulanur"],
    "ACME Shapar": ["acmeshapurgj", "shapur", "shapar"],
    "Cleanmax Honavad": ["cleanmaxhonavadka", "honavad"],
    "RenfraEnergy Trichy": ["renfratrichytn", "trichy"],
    "ReNew1_Gadag": ["renewgadagka", "gadag"],
    "Wanki": ["nslap", "wanki"],
}

tab_dashboard, tab_admin = st.tabs(["Dashboard", "Site Add-on"])

with tab_admin:
    st.subheader("Site Add-on")
    st.divider()
    st.markdown("## 1) Reference Excel (reference.xlsx)")
    if os.path.exists(REF_FILE_PATH):
        st.success(f"Reference file found: `{os.path.basename(REF_FILE_PATH)}`")
    else:
        st.warning("Reference file missing. Upload a reference Excel to enable power curve reference comparison.")

    ref_upload = st.file_uploader("Upload / Replace Reference Excel (.xlsx)", type=["xlsx"], key="ref_upload")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Save / Replace Reference Excel", type="primary"):
            if ref_upload is None:
                st.error("Please choose an .xlsx file first.")
            else:
                with open(REF_FILE_PATH, "wb") as f:
                    f.write(ref_upload.getbuffer())
                st.success("Reference Excel saved/replaced.")
                st.cache_data.clear()
                st.rerun()

    with c2:
        if st.button("Delete Reference Excel"):
            if os.path.exists(REF_FILE_PATH):
                os.remove(REF_FILE_PATH)
                st.success("Reference Excel deleted.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.info("No reference file to delete.")

with tab_dashboard:
    st.sidebar.subheader("Upload SCADA File")
    uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"], key="scada_upload")

    if uploaded_file is None:
        st.warning("Please upload SCADA file")
        st.stop()

    if not os.path.exists(REF_FILE_PATH):
        st.error("Reference Excel is missing. Upload `reference.xlsx` in the Admin tab.")
        st.stop()

    site = st.sidebar.selectbox("Select Site", list(SITE_CAPACITY.keys()), key="site_select")
    mode = st.sidebar.radio("Select View", ["Single Turbine", "Compare Turbines", "Show All Turbines"], key="mode_radio")

    @st.cache_data(show_spinner=True)
    def load_scada(file):
        chunksize = 200000
        chunks = pd.read_csv(file, chunksize=chunksize, low_memory=False, engine="c")
        df_local = pd.concat(chunks, ignore_index=True)
        df_local.columns = df_local.columns.str.strip()
        return df_local

    with st.spinner("Loading SCADA file..."):
        df = load_scada(uploaded_file)

    if "Name" not in df.columns:
        st.error("SCADA CSV must contain a 'Name' column for turbine identifier.")
        st.stop()

    available_columns = list(df.columns)

    auto_time = detect_column(available_columns, ["Time", "Timestamp", "DateTime", "LocalTime"])
    auto_wind = detect_column(available_columns, ["WindSpeedAve"])
    auto_power = detect_column(available_columns, [
        "PowerAve", "ActivePowerAve", "ActivePower", "Power",
        "GrdProdPwrAve", "GrdProdPwrAct", "WTGActivePowerAve", "OutputPowerAve"
    ])
    auto_pitch = detect_column(available_columns, ["BladePitchAve"])
    auto_nacelle = detect_column(available_columns, ["NacellePositionAve"])
    auto_pcurve = detect_column(available_columns, ["PCurveStsAve"])

    st.sidebar.markdown("### Column Mapping")

    time_col = st.sidebar.selectbox("Time Column", available_columns,
        index=available_columns.index(auto_time) if auto_time in available_columns else 0)
    wind_col = st.sidebar.selectbox("Wind Column", available_columns,
        index=available_columns.index(auto_wind) if auto_wind in available_columns else 0)
    power_col = st.sidebar.selectbox("Power Column", available_columns,
        index=available_columns.index(auto_power) if auto_power in available_columns else 0)
    pitch_col = st.sidebar.selectbox("Pitch Column", available_columns,
        index=available_columns.index(auto_pitch) if auto_pitch in available_columns else 0)
    nacelle_col = st.sidebar.selectbox("Nacelle Column", available_columns,
        index=available_columns.index(auto_nacelle) if auto_nacelle in available_columns else 0)
    pcurve_status_col = st.sidebar.selectbox("PCurve Status Column", available_columns,
        index=available_columns.index(auto_pcurve) if auto_pcurve in available_columns else 0)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
    df[power_col] = pd.to_numeric(df[power_col], errors="coerce")
    df[pitch_col] = pd.to_numeric(df[pitch_col], errors="coerce")
    df[nacelle_col] = pd.to_numeric(df[nacelle_col], errors="coerce")
    df[pcurve_status_col] = pd.to_numeric(df[pcurve_status_col], errors="coerce")
    df["Name"] = df["Name"].astype(str).str.strip()

    df = df.dropna(subset=["Name"])

    if df.empty:
        st.warning("SCADA file has no valid rows after parsing.")
        st.stop()

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
        st.warning("No SCADA data available for the selected date range.")
        st.stop()

    all_turbines = sorted(df["Name"].dropna().astype(str).str.strip().unique())
    num_turbines = len(all_turbines)
    capacity_per_turbine = SITE_CAPACITY.get(site, 3.3)
    total_capacity = num_turbines * capacity_per_turbine

    st.subheader(f"{site} | {num_turbines} Turbines | {capacity_per_turbine} MW Each | Total: {round(total_capacity, 2)} MW")
    st.markdown(f"Date Range: {start_day} → {end_day}")

    @st.cache_data
    def load_reference(site_name):
        try:
            ref_raw = pd.read_excel(REF_FILE_PATH, sheet_name="Power Curve", header=None)
        except Exception as e:
            st.error("Unable to open Power Curve sheet in reference.xlsx")
            st.code(str(e))
            st.stop()

        site_headers = ref_raw.iloc[2].copy()
        wind_series = pd.to_numeric(ref_raw.iloc[3:, 0], errors="coerce")
        normalized_site = normalize_text(site_name)
        aliases = [normalized_site] + [normalize_text(x) for x in SITE_REFERENCE_ALIASES.get(site_name, [])]

        matched_col = None
        matched_header = None
        for col_idx in range(1, ref_raw.shape[1]):
            header_text_raw = site_headers.iloc[col_idx]
            header_text = normalize_text(header_text_raw)
            if any(alias and alias in header_text for alias in aliases):
                matched_col = col_idx
                matched_header = str(header_text_raw)
                break

        if matched_col is None:
            st.error(f"Selected site '{site_name}' not found in reference.xlsx Power Curve sheet.")
            st.stop()

        ref_power = pd.to_numeric(ref_raw.iloc[3:, matched_col], errors="coerce")
        ref = pd.DataFrame({"WindSpeed": wind_series, "RefPower": ref_power}).dropna()
        ref = ref[(ref["WindSpeed"] >= 3) & (ref["WindSpeed"] <= 25)]

        if ref.empty:
            st.error(f"No valid reference data found for site '{site_name}'.")
            st.stop()

        wind_bins = np.arange(3, 25.5, BIN_SIZE)
        ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])
        return pd.DataFrame({"WindBin": wind_bins, "RefPower": ref_interp}), matched_header

    ref_curve, matched_reference_name = load_reference(site)
    st.caption(f"Matched Reference Curve: {matched_reference_name}")

    def process_turbine(t):
        df_t_all = df[df["Name"] == t].copy()
        raw_rows = len(df_t_all)

        result = {
            "turbine": t,
            "raw_rows": raw_rows,
            "status": "No Data",
            "comment": "No data available for this turbine in selected range.",
            "performance_comment": "Deviation not available",
            "nacelle_comment": "Nacelle average not available",
            "pcurve_comment": "PCurve status not available",
            "avg_nacelle": None,
            "avg_pcurve_status": None,
            "nacelle_band": "NA",
            "df_t": pd.DataFrame(columns=[wind_col, power_col, pitch_col, nacelle_col, pcurve_status_col]),
            "merged": ref_curve.copy(),
            "avg_dev": None
        }

        result["merged"]["AvgPower"] = np.nan
        result["merged"]["AvgPitch"] = np.nan

        if raw_rows == 0:
            return result

        df_t = df_t_all.dropna(subset=[wind_col, power_col, pitch_col, nacelle_col, pcurve_status_col]).copy()

        if df_t.empty:
            result["status"] = "No Valid Data"
            result["comment"] = "Rows exist, but required values are missing."
            return result

        df_t = df_t[
            (df_t[wind_col] >= 3) &
            (df_t[wind_col] <= 25) &
            (df_t[power_col] > 0) &
            (df_t[pitch_col] >= -5) &
            (df_t[pitch_col] <= 5) &
            (df_t[nacelle_col] >= 0) &
            (df_t[nacelle_col] <= 270) &
            (df_t[pcurve_status_col] >= 0)
        ].copy()

        if df_t.empty:
            result["status"] = "Filtered Out"
            result["comment"] = "Data exists, but all rows were removed by quality filters."
            return result

        result["avg_nacelle"] = df_t[nacelle_col].mean()
        result["nacelle_band"] = get_nacelle_band(result["avg_nacelle"])
        result["nacelle_comment"] = generate_nacelle_comment(result["avg_nacelle"])
        result["avg_pcurve_status"] = df_t[pcurve_status_col].mean()
        result["pcurve_comment"] = generate_pcurve_status_comment(result["avg_pcurve_status"])

        df_t["WindBin"] = (np.floor(df_t[wind_col] / BIN_SIZE) * BIN_SIZE).round(6)

        actual = df_t.groupby("WindBin").agg(
            AvgPower=(power_col, "mean"),
            AvgPitch=(pitch_col, "mean")
        ).reset_index()

        merged = ref_curve.merge(actual, on="WindBin", how="left")
        merged["AvgPower"] = safe_savgol(merged["AvgPower"], window=7, poly=2)
        merged["Deviation_%"] = np.where(
            merged["RefPower"] > 0,
            ((merged["AvgPower"] - merged["RefPower"]) / merged["RefPower"]) * 100,
            np.nan
        )

        avg_dev = merged["Deviation_%"].mean(skipna=True)
        if pd.isna(avg_dev):
            avg_dev = None

        result["df_t"] = df_t
        result["merged"] = merged
        result["avg_dev"] = avg_dev

        if avg_dev is None:
            result["status"] = "Low Data"
            result["performance_comment"] = "Valid turbine rows exist, but insufficient overlap with reference bins."
        else:
            result["status"] = "OK"
            result["performance_comment"] = generate_performance_comment(avg_dev)

        result["comment"] = f"{result['performance_comment']} | {result['nacelle_comment']} | {result['pcurve_comment']}"
        return result

    def plot_graph(df_t, merged, title, dev, avg_nacelle, avg_pcurve_status, nacelle_comment, perf_comment, pcurve_comment):
        safe_dev = 0 if dev is None or pd.isna(dev) else dev
        title_color = "green" if -2 <= safe_dev <= 2 else ("orange" if safe_dev < -2 else "red")
        dev_txt = "NA" if dev is None or pd.isna(dev) else round(dev, 2)
        nacelle_txt = "NA" if avg_nacelle is None or pd.isna(avg_nacelle) else round(avg_nacelle, 2)
        pcurve_txt = "NA" if avg_pcurve_status is None or pd.isna(avg_pcurve_status) else round(avg_pcurve_status, 2)

        fig = go.Figure()

        if not df_t.empty:
            fig.add_trace(go.Scatter(
                x=df_t[wind_col], y=df_t[power_col], mode="markers",
                marker=dict(size=4, opacity=0.30, color="rgba(30, 144, 255, 0.45)"),
                name="Power Scatter"
            ))
            fig.add_trace(go.Scatter(
                x=df_t[wind_col], y=df_t[pitch_col], mode="markers",
                marker=dict(size=4, opacity=0.25, color="rgba(255, 165, 0, 0.35)"),
                name="Pitch Scatter", yaxis="y2"
            ))

        fig.add_trace(go.Scatter(
            x=merged["WindBin"], y=merged["RefPower"], mode="lines",
            line=dict(dash="dash", width=3, color="red"), name="Reference Power"
        ))

        if merged["AvgPower"].notna().any():
            fig.add_trace(go.Scatter(
                x=merged["WindBin"], y=merged["AvgPower"], mode="lines+markers",
                line=dict(width=4, color="green"), marker=dict(size=6, color="green"),
                name="Actual Avg Power"
            ))

        if merged["AvgPitch"].notna().any():
            fig.add_trace(go.Scatter(
                x=merged["WindBin"], y=merged["AvgPitch"], mode="lines+markers",
                line=dict(width=3, color="orange"), marker=dict(size=5, color="orange"),
                name="Avg Blade Pitch", yaxis="y2"
            ))

        fig.update_layout(
            title=dict(
                text=f"{title} | Dev: {dev_txt}% | NacelleAvg: {nacelle_txt}° | PCurveStsAve: {pcurve_txt}",
                font=dict(color=title_color)
            ),
            xaxis=dict(title="Wind Speed"),
            yaxis=dict(title="Power"),
            yaxis2=dict(title="Blade Pitch", overlaying="y", side="right", showgrid=False),
            height=580,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            annotations=[
                dict(text=perf_comment, x=0.5, y=0.97, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="gray")),
                dict(text=nacelle_comment, x=0.5, y=0.92, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="gray")),
                dict(text=pcurve_comment, x=0.5, y=0.87, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="gray")),
            ]
        )
        return fig

    if mode == "Single Turbine":
        turbines_to_show = [st.sidebar.selectbox("Select Turbine", all_turbines, key="single_turbine")]
    elif mode == "Compare Turbines":
        turbines_to_show = st.sidebar.multiselect("Select Turbines", all_turbines, default=all_turbines, key="compare_turbines")
    else:
        turbines_to_show = all_turbines

    if not turbines_to_show:
        st.warning("Please select at least one turbine.")
        st.stop()

    cols = st.columns(2)
    results = []
    figures = []

    for i, t in enumerate(turbines_to_show):
        res = process_turbine(t)
        with cols[i % 2]:
            fig = plot_graph(
                res["df_t"], res["merged"], res["turbine"], res["avg_dev"],
                res["avg_nacelle"], res["avg_pcurve_status"],
                res["nacelle_comment"], res["performance_comment"], res["pcurve_comment"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### Analysis")
            st.code(res["comment"])

        figures.append((t, fig, res["comment"]))
        results.append({
            "Turbine": t,
            "Deviation_%": None if res["avg_dev"] is None else round(res["avg_dev"], 2),
            "Avg_Nacelle": None if res["avg_nacelle"] is None else round(res["avg_nacelle"], 2),
            "Nacelle_Band": res["nacelle_band"],
            "PCurveStsAve": None if res["avg_pcurve_status"] is None else round(res["avg_pcurve_status"], 2),
            "Status": res["status"],
            "Rows": res["raw_rows"],
            "Comment": res["comment"]
        })

    st.subheader("Turbine Ranking")
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(by="Deviation_%", na_position="last")

        def color_row(row):
            if row["Status"] == "OK":
                return ['background-color: #ccffcc'] * len(row)
            elif row["Status"] == "Low Data":
                return ['background-color: #d9d9d9'] * len(row)
            elif row["Status"] == "Filtered Out":
                return ['background-color: #ffe0b3'] * len(row)
            elif row["Status"] in ["No Valid Data", "No Data"]:
                return ['background-color: #f2f2f2'] * len(row)
            return ['background-color: #cccccc'] * len(row)

        st.dataframe(results_df.style.apply(color_row, axis=1), use_container_width=True)

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
                pcs_text = "NA" if pd.isna(row["PCurveStsAve"]) else row["PCurveStsAve"]
                line = f"{row['Turbine']} | Dev: {dev_text}% | Nacelle: {nac_text}° | Band: {row['Nacelle_Band']} | PCurveStsAve: {pcs_text} | {row['Status']}"
                pdf.drawString(40, y, line[:150])
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
            mime="application/pdf"
        )

    except Exception as e:
        st.error("PDF generation failed")
        st.code(str(e))
