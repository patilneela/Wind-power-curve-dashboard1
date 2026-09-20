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


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Power Curve Analytics Report",
    layout="wide"
)


# ============================================================
# KALEIDO CHECK
# ============================================================
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(__file__)

REF_FILE_PATH = os.path.join(BASE_DIR, "reference.xlsx")

SITE_MASTER_XLSX = os.path.join(BASE_DIR, "site_master.xlsx")
SITE_MASTER_CSV = os.path.join(BASE_DIR, "site_master.csv")

BIN_SIZE = 0.5


# ============================================================
# LOGIN
# ============================================================
def login_gate():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("Power Curve Analytics Report")
    st.subheader("Login")

    try:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):

            try:
                auth = st.secrets["auth"]

                valid_user = (
                    username == auth["username"]
                    and password == auth["password"]
                )

            except Exception:
                valid_user = False

            if valid_user:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password.")

        return False

    except Exception:
        st.warning(
            "Authentication settings are not configured. "
            "Please configure [auth] in Streamlit secrets."
        )

        if st.button("Continue"):
            st.session_state.authenticated = True
            st.rerun()

        return False


if not login_gate():
    st.stop()


# ============================================================
# SIDEBAR LOGOUT
# ============================================================
with st.sidebar:

    st.markdown("### Power Curve Dashboard")

    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================
def normalize_column_name(value):

    value = str(value).strip().lower()

    for char in [" ", "_", "-", "/", "\\", ".", "(", ")", "[", "]"]:
        value = value.replace(char, "")

    return value


# ============================================================
# FIND COLUMN
# ============================================================
def find_column(columns, possible_names):

    normalized = {
        normalize_column_name(col): col
        for col in columns
    }

    # Exact match
    for name in possible_names:

        key = normalize_column_name(name)

        if key in normalized:
            return normalized[key]

    # Contains match
    for col in columns:

        ncol = normalize_column_name(col)

        for name in possible_names:

            nname = normalize_column_name(name)

            if nname in ncol:
                return col

    return None


# ============================================================
# READ REFERENCE EXCEL AND AUTOMATICALLY FIND HEADER
# ============================================================
@st.cache_data(show_spinner=False)
def load_new_reference(uploaded_file_bytes=None):

    try:

        # ----------------------------------------------------
        # Determine file
        # ----------------------------------------------------
        if uploaded_file_bytes is not None:

            excel_data = io.BytesIO(uploaded_file_bytes)

        else:

            if not os.path.exists(REF_FILE_PATH):
                return None, None, "Reference Excel file not found."

            excel_data = REF_FILE_PATH

        # ----------------------------------------------------
        # Read all sheets without assuming header
        # ----------------------------------------------------
        sheets = pd.read_excel(
            excel_data,
            sheet_name=None,
            header=None
        )

        best_df = None
        best_score = -1
        best_sheet = None
        best_header_row = None

        # ----------------------------------------------------
        # Search every sheet and every row for header
        # ----------------------------------------------------
        for sheet_name, raw_df in sheets.items():

            if raw_df.empty:
                continue

            max_rows_to_check = min(len(raw_df), 100)

            for row_idx in range(max_rows_to_check):

                row_values = [
                    str(x).strip().lower()
                    for x in raw_df.iloc[row_idx].tolist()
                ]

                row_values = [
                    x for x in row_values
                    if x not in ["", "nan", "none"]
                ]

                if not row_values:
                    continue

                has_site = any(
                    (
                        "site" in x
                        or "plant" in x
                        or "project" in x
                        or "windfarm" in x
                        or "windfarmname" in x
                    )
                    for x in row_values
                )

                has_turbine = any(
                    (
                        "turbine" in x
                        or "wtg" in x
                        or x in ["name", "turbine name", "turbine_name"]
                        or "turbineid" in x
                    )
                    for x in row_values
                )

                score = 0

                if has_site:
                    score += 10

                if has_turbine:
                    score += 10

                if has_site and has_turbine:
                    score += 20

                if score > best_score:

                    best_score = score
                    best_df = raw_df.copy()
                    best_sheet = sheet_name
                    best_header_row = row_idx

        # ----------------------------------------------------
        # Header not found
        # ----------------------------------------------------
        if best_df is None or best_score < 20:

            return (
                None,
                None,
                "Could not automatically find a row containing "
                "both Site and Turbine/Name/WTG columns."
            )

        # ----------------------------------------------------
        # Apply detected header
        # ----------------------------------------------------
        header = best_df.iloc[best_header_row].astype(str).str.strip()

        df = best_df.iloc[best_header_row + 1:].copy()

        df.columns = header

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # Remove completely empty rows
        df = df.dropna(axis=0, how="all")

        # ----------------------------------------------------
        # Detect Site column
        # ----------------------------------------------------
        site_col = find_column(
            df.columns,
            [
                "Site",
                "Site Name",
                "Site_Name",
                "SiteName",
                "Plant",
                "Plant Name",
                "Project",
                "Project Name",
                "Wind Farm",
                "Wind Farm Name",
                "WindFarm"
            ]
        )

        # ----------------------------------------------------
        # Detect Turbine column
        # ----------------------------------------------------
        turbine_col = find_column(
            df.columns,
            [
                "Turbine",
                "Turbine Name",
                "Turbine_Name",
                "TurbineName",
                "Turbine ID",
                "Turbine_ID",
                "TurbineID",
                "Name",
                "WTG",
                "WTG Name",
                "WTG_Name",
                "WTGName",
                "WTG ID",
                "Asset",
                "Asset Name"
            ]
        )

        if site_col is None:

            return (
                None,
                None,
                f"Site column could not be detected.\n\n"
                f"Detected columns:\n{list(df.columns)}"
            )

        if turbine_col is None:

            return (
                None,
                None,
                f"Turbine/Name/WTG column could not be detected.\n\n"
                f"Detected columns:\n{list(df.columns)}"
            )

        # ----------------------------------------------------
        # Keep only useful columns
        # ----------------------------------------------------
        mapping = df[[site_col, turbine_col]].copy()

        mapping.columns = ["Site", "Turbine"]

        # ----------------------------------------------------
        # Clean values
        # ----------------------------------------------------
        mapping["Site"] = (
            mapping["Site"]
            .astype(str)
            .str.strip()
        )

        mapping["Turbine"] = (
            mapping["Turbine"]
            .astype(str)
            .str.strip()
        )

        # Remove invalid values
        invalid_values = [
            "",
            "nan",
            "none",
            "null",
            "nat"
        ]

        mapping = mapping[
            ~mapping["Site"].str.lower().isin(invalid_values)
            &
            ~mapping["Turbine"].str.lower().isin(invalid_values)
        ]

        # ----------------------------------------------------
        # Normalize matching names
        # ----------------------------------------------------
        mapping["Site_Normalized"] = (
            mapping["Site"]
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        mapping["Turbine_Normalized"] = (
            mapping["Turbine"]
            .str.lower()
            .str.replace(r"\s+", "", regex=True)
            .str.replace("_", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.strip()
        )

        # Remove duplicate mappings
        mapping = mapping.drop_duplicates(
            subset=["Site_Normalized", "Turbine_Normalized"]
        )

        mapping = mapping.reset_index(drop=True)

        info = {
            "sheet": best_sheet,
            "header_row": best_header_row,
            "site_col": site_col,
            "turbine_col": turbine_col,
            "all_columns": list(df.columns)
        }

        return mapping, info, None

    except Exception as e:

        return None, None, f"Reference Excel error: {e}"


# ============================================================
# LOAD SITE MASTER
# ============================================================
@st.cache_data(show_spinner=False)
def load_site_master():

    site_capacity = {}

    try:

        if os.path.exists(SITE_MASTER_XLSX):

            df = pd.read_excel(SITE_MASTER_XLSX)

        elif os.path.exists(SITE_MASTER_CSV):

            df = pd.read_csv(SITE_MASTER_CSV)

        else:

            return site_capacity

        site_col = find_column(
            df.columns,
            [
                "Site",
                "Site Name",
                "Site_Name",
                "SiteName",
                "Plant"
            ]
        )

        capacity_col = find_column(
            df.columns,
            [
                "Capacity",
                "Capacity MW",
                "MW",
                "Rated Capacity",
                "Turbine Capacity"
            ]
        )

        if site_col is None:
            return site_capacity

        for _, row in df.iterrows():

            site = str(row[site_col]).strip()

            if not site or site.lower() == "nan":
                continue

            capacity = 3.3

            if capacity_col is not None:

                try:
                    capacity = float(row[capacity_col])
                except Exception:
                    capacity = 3.3

            site_capacity[site] = capacity

    except Exception:
        pass

    return site_capacity


SITE_CAPACITY = load_site_master()


# ============================================================
# FALLBACK CAPACITY
# ============================================================
DEFAULT_CAPACITY = 3.3


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## Input Files")


# ============================================================
# SCADA UPLOAD
# ============================================================
scada_file = st.sidebar.file_uploader(
    "1. Upload SCADA CSV",
    type=["csv"]
)


# ============================================================
# NEW REFERENCE UPLOAD
# ============================================================
reference_upload = st.sidebar.file_uploader(
    "2. Upload New Reference Excel",
    type=["xlsx", "xls"]
)


# ============================================================
# LOAD REFERENCE
# ============================================================
reference_mapping = None
reference_info = None
reference_error = None


if reference_upload is not None:

    reference_mapping, reference_info, reference_error = (
        load_new_reference(
            reference_upload.getvalue()
        )
    )

else:

    if os.path.exists(REF_FILE_PATH):

        reference_mapping, reference_info, reference_error = (
            load_new_reference()
        )


# ============================================================
# REFERENCE STATUS
# ============================================================
if reference_error:

    st.error(reference_error)

    st.info(
        "Your new Excel must contain a header row with "
        "a Site column and a Turbine/Name/WTG column."
    )

    st.stop()


if reference_mapping is None:

    st.warning(
        "Please upload the new Reference Excel file."
    )

    st.stop()


# ============================================================
# REFERENCE INFO
# ============================================================
with st.sidebar.expander("Reference File Information"):

    st.write(
        f"Sheet: **{reference_info['sheet']}**"
    )

    st.write(
        f"Header row: **{reference_info['header_row'] + 1}**"
    )

    st.write(
        f"Site column: **{reference_info['site_col']}**"
    )

    st.write(
        f"Turbine column: **{reference_info['turbine_col']}**"
    )

    st.write(
        f"Reference turbine records: "
        f"**{len(reference_mapping)}**"
    )


# ============================================================
# SITE LIST FROM NEW REFERENCE
# ============================================================
reference_sites = sorted(
    reference_mapping["Site"].dropna().unique().tolist()
)


if not reference_sites:

    st.error(
        "No sites were found in the new reference Excel."
    )

    st.stop()


# ============================================================
# SITE SELECTION
# ============================================================
st.sidebar.markdown("---")

site = st.sidebar.selectbox(
    "Select Site",
    reference_sites
)


# ============================================================
# GET TURBINES FOR SELECTED SITE
# ============================================================
site_mapping = reference_mapping[
    reference_mapping["Site_Normalized"]
    ==
    site.lower().strip()
].copy()


reference_turbines = (
    site_mapping["Turbine"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)


# ============================================================
# SHOW REFERENCE TURBINE COUNT
# ============================================================
st.sidebar.info(
    f"Reference turbines for {site}: "
    f"**{len(reference_turbines)}**"
)


# ============================================================
# VIEW MODE
# ============================================================
view_mode = st.sidebar.selectbox(
    "Select View",
    [
        "Single Turbine",
        "Compare Turbines",
        "Show All Turbines"
    ]
)


# ============================================================
# LOAD SCADA
# ============================================================
@st.cache_data(show_spinner=False)
def load_scada(uploaded_bytes):

    try:

        all_chunks = []

        for chunk in pd.read_csv(
            io.BytesIO(uploaded_bytes),
            chunksize=200000,
            low_memory=False
        ):

            chunk.columns = [
                str(c).strip()
                for c in chunk.columns
            ]

            all_chunks.append(chunk)

        if not all_chunks:

            return None, None

        df = pd.concat(
            all_chunks,
            ignore_index=True
        )

        # ----------------------------------------------------
        # Turbine column
        # ----------------------------------------------------
        turbine_col = find_column(
            df.columns,
            [
                "Name",
                "Turbine",
                "Turbine Name",
                "Turbine_Name",
                "TurbineName",
                "WTG",
                "WTG Name",
                "WTG_Name",
                "WTGName"
            ]
        )

        # ----------------------------------------------------
        # Wind
        # ----------------------------------------------------
        wind_col = find_column(
            df.columns,
            [
                "WindSpeedAve",
                "Wind Speed Ave",
                "WindSpeed",
                "Wind Speed",
                "Average Wind Speed"
            ]
        )

        # ----------------------------------------------------
        # Power
        # ----------------------------------------------------
        power_col = find_column(
            df.columns,
            [
                "ActivePWAve",
                "Active Power Ave",
                "ActivePower",
                "Active Power",
                "Power",
                "PowerAve",
                "ActivePwrAve"
            ]
        )

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------
        time_col = find_column(
            df.columns,
            [
                "Time",
                "Timestamp",
                "DateTime",
                "Date Time",
                "Date",
                "TimeStamp",
                "Time UTC"
            ]
        )

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------
        pitch_col = find_column(
            df.columns,
            [
                "BldPitch1Ave",
                "BldPitch",
                "Pitch",
                "PitchAve",
                "Blade Pitch"
            ]
        )

        missing = []

        if turbine_col is None:
            missing.append("Turbine/Name/WTG")

        if wind_col is None:
            missing.append("Wind Speed")

        if power_col is None:
            missing.append("Power")

        if time_col is None:
            missing.append("Time")

        if missing:

            return (
                None,
                "Could not detect SCADA columns: "
                + ", ".join(missing)
            )

        # ----------------------------------------------------
        # Rename to standard names
        # ----------------------------------------------------
        rename_dict = {
            turbine_col: "Name",
            wind_col: "WindSpeed",
            power_col: "ActivePower",
            time_col: "Timestamp"
        }

        if pitch_col is not None:
            rename_dict[pitch_col] = "Pitch"

        df = df.rename(
            columns=rename_dict
        )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------
        df["Name"] = (
            df["Name"]
            .astype(str)
            .str.strip()
        )

        df["WindSpeed"] = pd.to_numeric(
            df["WindSpeed"],
            errors="coerce"
        )

        df["ActivePower"] = pd.to_numeric(
            df["ActivePower"],
            errors="coerce"
        )

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        if "Pitch" in df.columns:

            df["Pitch"] = pd.to_numeric(
                df["Pitch"],
                errors="coerce"
            )

        df = df.dropna(
            subset=[
                "Name",
                "WindSpeed",
                "ActivePower",
                "Timestamp"
            ]
        )

        return df, None

    except Exception as e:

        return None, str(e)


# ============================================================
# STOP IF NO SCADA
# ============================================================
if scada_file is None:

    st.title("Power Curve Analytics Report")

    st.info(
        "Upload the SCADA CSV to generate the power curves."
    )

    st.stop()


# ============================================================
# LOAD SCADA
# ============================================================
df, scada_error = load_scada(
    scada_file.getvalue()
)


if scada_error:

    st.error(scada_error)
    st.stop()


if df is None or df.empty:

    st.error(
        "No SCADA data could be loaded."
    )

    st.stop()


# ============================================================
# NORMALIZE SCADA TURBINE NAMES
# ============================================================
df["Name_Normalized"] = (
    df["Name"]
    .astype(str)
    .str.lower()
    .str.replace(r"\s+", "", regex=True)
    .str.replace("_", "", regex=False)
    .str.replace("-", "", regex=False)
    .str.strip()
)


# ============================================================
# DATE FILTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## Date Range")


max_scada_date = df["Timestamp"].max()

date_option = st.sidebar.selectbox(
    "Date Option",
    [
        "Clear",
        "Today",
        "This Week",
        "This Month",
        "Last Week",
        "Last Month",
        "Manual Calendar"
    ]
)


# ============================================================
# DATE RANGE CALCULATION
# ============================================================
if date_option == "Clear":

    start_date = (
        max_scada_date
        - timedelta(days=15)
    ).normalize()

    end_date = max_scada_date.normalize()


elif date_option == "Today":

    start_date = max_scada_date.normalize()
    end_date = max_scada_date.normalize()


elif date_option == "This Week":

    start_date = (
        max_scada_date
        - timedelta(days=max_scada_date.weekday())
    ).normalize()

    end_date = max_scada_date.normalize()


elif date_option == "This Month":

    start_date = max_scada_date.replace(
        day=1
    ).normalize()

    end_date = max_scada_date.normalize()


elif date_option == "Last Week":

    this_monday = (
        max_scada_date
        - timedelta(days=max_scada_date.weekday())
    ).normalize()

    start_date = this_monday - timedelta(days=7)
    end_date = this_monday - timedelta(days=1)


elif date_option == "Last Month":

    first_this_month = max_scada_date.replace(
        day=1
    ).normalize()

    end_date = first_this_month - timedelta(days=1)

    start_date = end_date.replace(
        day=1
    )


else:

    min_date = df["Timestamp"].min().date()
    max_date = df["Timestamp"].max().date()

    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, tuple):

        if len(selected_dates) == 2:

            start_date = pd.Timestamp(
                selected_dates[0]
            )

            end_date = pd.Timestamp(
                selected_dates[1]
            )

        else:

            start_date = pd.Timestamp(
                selected_dates[0]
            )

            end_date = start_date

    else:

        start_date = pd.Timestamp(
            selected_dates
        )

        end_date = start_date


# ============================================================
# APPLY DATE FILTER
# ============================================================
end_datetime = (
    end_date
    + timedelta(days=1)
    - timedelta(microseconds=1)
)

df_date = df[
    (df["Timestamp"] >= start_date)
    &
    (df["Timestamp"] <= end_datetime)
].copy()


# ============================================================
# FILTER BASED ON NEW REFERENCE FILE
# ============================================================
reference_turbines_normalized = set(
    site_mapping["Turbine_Normalized"]
    .dropna()
    .tolist()
)


df_site = df_date[
    df_date["Name_Normalized"].isin(
        reference_turbines_normalized
    )
].copy()


# ============================================================
# DATA TURBINES PRESENT
# ============================================================
data_turbines = (
    df_site["Name"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)


# ============================================================
# MAP NORMALIZED SCADA NAMES
# ============================================================
scada_name_map = {}

for name in data_turbines:

    normalized = (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    scada_name_map[normalized] = name


# ============================================================
# AVAILABLE REFERENCE TURBINES
# ============================================================
available_reference_turbines = []

missing_reference_turbines = []

for ref_turbine in reference_turbines:

    normalized = (
        str(ref_turbine)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    if normalized in scada_name_map:

        available_reference_turbines.append(
            scada_name_map[normalized]
        )

    else:

        missing_reference_turbines.append(
            ref_turbine
        )


# ============================================================
# HEADER
# ============================================================
reference_count = len(reference_turbines)
available_count = len(available_reference_turbines)

capacity_per_turbine = SITE_CAPACITY.get(
    site,
    DEFAULT_CAPACITY
)

total_capacity = (
    reference_count
    * capacity_per_turbine
)


st.title("Power Curve Analytics Report")


st.subheader(
    f"{site} | "
    f"{reference_count} Reference Turbines | "
    f"{available_count} With SCADA Data | "
    f"{capacity_per_turbine} MW Each | "
    f"Total: {round(total_capacity, 2)} MW"
)


st.caption(
    f"Applied Range: "
    f"{start_date.strftime('%Y-%m-%d')} → "
    f"{end_date.strftime('%Y-%m-%d')}"
)


# ============================================================
# DEBUG INFORMATION
# ============================================================
with st.expander("Reference / SCADA Matching Details"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"**Reference turbines: {reference_count}**"
        )

        st.dataframe(
            pd.DataFrame({
                "Reference Turbine": reference_turbines
            }),
            use_container_width=True,
            height=250
        )

    with col2:

        st.markdown(
            f"**SCADA turbines found: {available_count}**"
        )

        st.dataframe(
            pd.DataFrame({
                "SCADA Turbine": available_reference_turbines
            }),
            use_container_width=True,
            height=250
        )

    if missing_reference_turbines:

        st.warning(
            f"{len(missing_reference_turbines)} reference turbines "
            f"do not have SCADA data in the selected date range."
        )

        st.dataframe(
            pd.DataFrame({
                "Reference Turbine Without SCADA":
                    missing_reference_turbines
            }),
            use_container_width=True
        )


# ============================================================
# POWER CURVE PROCESSING
# ============================================================
def process_turbine(turbine_df):

    df_t = turbine_df.copy()

    # --------------------------------------------------------
    # Valid SCADA range
    # --------------------------------------------------------
    df_t = df_t[
        (df_t["WindSpeed"] >= 3)
        &
        (df_t["WindSpeed"] <= 25)
        &
        (df_t["ActivePower"] > 0)
    ].copy()

    # --------------------------------------------------------
    # Pitch filter
    # --------------------------------------------------------
    if "Pitch" in df_t.columns:

        df_t = df_t[
            (df_t["Pitch"] >= -5)
            &
            (df_t["Pitch"] <= 5)
        ].copy()

    if len(df_t) < 30:

        return None, None, {
            "Data Points": len(df_t),
            "Curve Points": 0,
            "Avg Power": np.nan,
            "Max Power": np.nan,
            "Min Wind": np.nan,
            "Max Wind": np.nan,
            "Status": "Insufficient Data"
        }

    # --------------------------------------------------------
    # Wind bins
    # --------------------------------------------------------
    df_t["WindBin"] = (
        np.floor(
            df_t["WindSpeed"] / BIN_SIZE
        )
        * BIN_SIZE
    )

    # --------------------------------------------------------
    # Average power per wind bin
    # --------------------------------------------------------
    curve = (
        df_t
        .groupby("WindBin", as_index=False)
        ["ActivePower"]
        .mean()
        .rename(
            columns={
                "ActivePower": "AvgPower"
            }
        )
        .sort_values("WindBin")
    )

    curve = curve.dropna()

    if curve.empty:

        return None, None, {
            "Data Points": len(df_t),
            "Curve Points": 0,
            "Avg Power": np.nan,
            "Max Power": np.nan,
            "Min Wind": np.nan,
            "Max Wind": np.nan,
            "Status": "No Curve"
        }

    # --------------------------------------------------------
    # Smooth curve
    # --------------------------------------------------------
    if len(curve) >= 7:

        window = 7

        if window > len(curve):
            window = len(curve)

        if window % 2 == 0:
            window -= 1

        if window >= 5:

            try:

                curve["SmoothPower"] = savgol_filter(
                    curve["AvgPower"].values,
                    window_length=window,
                    polyorder=2
                )

            except Exception:

                curve["SmoothPower"] = curve["AvgPower"]

        else:

            curve["SmoothPower"] = curve["AvgPower"]

    else:

        curve["SmoothPower"] = curve["AvgPower"]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metrics = {
        "Data Points": len(df_t),
        "Curve Points": len(curve),
        "Avg Power": df_t["ActivePower"].mean(),
        "Max Power": df_t["ActivePower"].max(),
        "Min Wind": df_t["WindSpeed"].min(),
        "Max Wind": df_t["WindSpeed"].max(),
        "Status": "Available"
    }

    return df_t, curve, metrics


# ============================================================
# GRAPH FUNCTION
# ============================================================
def create_power_curve_plot(
    turbine_name,
    turbine_df,
    curve
):

    fig = go.Figure()

    # --------------------------------------------------------
    # Raw SCADA points
    # --------------------------------------------------------
    fig.add_trace(
        go.Scattergl(
            x=turbine_df["WindSpeed"],
            y=turbine_df["ActivePower"],
            mode="markers",
            name="SCADA Data",
            marker=dict(
                size=5,
                opacity=0.35
            ),
            hovertemplate=(
                "<b>SCADA Point</b><br>"
                "Wind Speed: %{x:.2f} m/s<br>"
                "Power: %{y:.2f}<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # Average power curve
    # --------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=curve["WindBin"],
            y=curve["SmoothPower"],
            mode="lines+markers",
            name="Power Curve",
            line=dict(
                width=3
            ),
            marker=dict(
                size=7
            ),
            hovertemplate=(
                "<b>Power Curve</b><br>"
                "Wind Speed: %{x:.1f} m/s<br>"
                "Average Power: %{y:.2f}<extra></extra>"
            )
        )
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    fig.update_layout(
        title=dict(
            text=f"{turbine_name} — Power Curve",
            x=0.5,
            xanchor="center"
        ),

        xaxis=dict(
            title="Wind Speed (m/s)",
            range=[
                max(0, float(turbine_df["WindSpeed"].min()) - 1),
                min(26, float(turbine_df["WindSpeed"].max()) + 1)
            ],
            dtick=1,
            showgrid=True,
            zeroline=False
        ),

        yaxis=dict(
            title="Power",
            showgrid=True,
            zeroline=False
        ),

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        height=550,

        margin=dict(
            l=70,
            r=40,
            t=90,
            b=70
        )
    )

    return fig


# ============================================================
# PROCESS ALL REFERENCE TURBINES
# ============================================================
results = {}

for turbine in reference_turbines:

    turbine_normalized = (
        str(turbine)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    turbine_data = df_site[
        df_site["Name_Normalized"]
        == turbine_normalized
    ].copy()

    if turbine_data.empty:

        results[turbine] = {
            "df": None,
            "curve": None,
            "metrics": {
                "Data Points": 0,
                "Curve Points": 0,
                "Avg Power": np.nan,
                "Max Power": np.nan,
                "Min Wind": np.nan,
                "Max Wind": np.nan,
                "Status": "No SCADA Data"
            }
        }

        continue

    df_t, curve, metrics = process_turbine(
        turbine_data
    )

    results[turbine] = {
        "df": df_t,
        "curve": curve,
        "metrics": metrics
    }


# ============================================================
# AVAILABLE CURVES
# ============================================================
valid_turbines = [
    turbine
    for turbine in reference_turbines
    if results[turbine]["curve"] is not None
]


# ============================================================
# SUMMARY TABLE
# ============================================================
st.markdown("---")
st.subheader("Turbine Summary")


summary_rows = []

for turbine in reference_turbines:

    metrics = results[turbine]["metrics"]

    summary_rows.append({
        "Turbine": turbine,
        "Data Points": metrics["Data Points"],
        "Curve Points": metrics["Curve Points"],
        "Min Wind (m/s)": (
            round(metrics["Min Wind"], 2)
            if pd.notna(metrics["Min Wind"])
            else "-"
        ),
        "Max Wind (m/s)": (
            round(metrics["Max Wind"], 2)
            if pd.notna(metrics["Max Wind"])
            else "-"
        ),
        "Average Power": (
            round(metrics["Avg Power"], 2)
            if pd.notna(metrics["Avg Power"])
            else "-"
        ),
        "Maximum Power": (
            round(metrics["Max Power"], 2)
            if pd.notna(metrics["Max Power"])
            else "-"
        ),
        "Status": metrics["Status"]
    })


summary_df = pd.DataFrame(summary_rows)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# NO VALID CURVES
# ============================================================
if not valid_turbines:

    st.error(
        "No valid power curves could be generated for the "
        "selected site and date range."
    )

    st.info(
        "Check the Reference/SCADA Matching Details above. "
        "The reference turbine names must match the SCADA "
        "Name column."
    )

    st.stop()


# ============================================================
# SINGLE TURBINE
# ============================================================
if view_mode == "Single Turbine":

    st.markdown("---")
    st.subheader("Single Turbine Power Curve")

    selected_turbine = st.selectbox(
        "Select Turbine",
        reference_turbines
    )

    result = results[selected_turbine]

    if result["curve"] is None:

        st.warning(
            f"No valid SCADA data available for "
            f"**{selected_turbine}** in the selected date range."
        )

    else:

        fig = create_power_curve_plot(
            selected_turbine,
            result["df"],
            result["curve"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True
            }
        )


# ============================================================
# COMPARE TURBINES
# ============================================================
elif view_mode == "Compare Turbines":

    st.markdown("---")
    st.subheader("Compare Turbine Power Curves")

    selected_turbines = st.multiselect(
        "Select Turbines",
        reference_turbines,
        default=valid_turbines[:min(3, len(valid_turbines))]
    )

    if not selected_turbines:

        st.info(
            "Select at least one turbine."
        )

    else:

        fig = go.Figure()

        for turbine in selected_turbines:

            result = results[turbine]

            if result["curve"] is None:
                continue

            curve = result["curve"]

            fig.add_trace(
                go.Scatter(
                    x=curve["WindBin"],
                    y=curve["SmoothPower"],
                    mode="lines+markers",
                    name=turbine,
                    marker=dict(
                        size=6
                    ),
                    hovertemplate=(
                        f"<b>{turbine}</b><br>"
                        "Wind Speed: %{x:.1f} m/s<br>"
                        "Average Power: %{y:.2f}"
                        "<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            title=dict(
                text=f"{site} — Turbine Power Curve Comparison",
                x=0.5,
                xanchor="center"
            ),

            xaxis=dict(
                title="Wind Speed (m/s)",
                dtick=1,
                showgrid=True
            ),

            yaxis=dict(
                title="Power",
                showgrid=True
            ),

            hovermode="closest",

            height=600,

            legend=dict(
                orientation="v",
                x=1.02,
                y=1
            ),

            margin=dict(
                l=70,
                r=180,
                t=90,
                b=70
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True
            }
        )


# ============================================================
# SHOW ALL TURBINES
# ============================================================
else:

    st.markdown("---")
    st.subheader(
        f"All Turbine Power Curves — {site}"
    )

    for turbine in reference_turbines:

        result = results[turbine]

        with st.expander(
            f"{turbine} | {result['metrics']['Status']}",
            expanded=False
        ):

            if result["curve"] is None:

                st.warning(
                    "No valid power curve available for "
                    "this turbine."
                )

                continue

            fig = create_power_curve_plot(
                turbine,
                result["df"],
                result["curve"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True
                }
            )


# ============================================================
# PDF REPORT
# ============================================================
st.markdown("---")
st.subheader("Report")


def create_pdf_report(
    site_name,
    summary,
    results_dict
):

    if not KALEIDO_AVAILABLE:

        return None

    pdf_buffer = io.BytesIO()

    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=landscape(A4)
    )

    width, height = landscape(A4)

    # --------------------------------------------------------
    # Cover
    # --------------------------------------------------------
    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawCentredString(
        width / 2,
        height - 50,
        "Power Curve Analytics Report"
    )

    pdf.setFont(
        "Helvetica",
        13
    )

    pdf.drawCentredString(
        width / 2,
        height - 75,
        site_name
    )

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawCentredString(
        width / 2,
        height - 95,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    )

    pdf.showPage()

    # --------------------------------------------------------
    # Turbine graphs
    # --------------------------------------------------------
    for turbine in reference_turbines:

        result = results_dict.get(turbine)

        if not result:
            continue

        if result["curve"] is None:
            continue

        try:

            fig = create_power_curve_plot(
                turbine,
                result["df"],
                result["curve"]
            )

            image_bytes = fig.to_image(
                format="png",
                width=1400,
                height=750,
                scale=1.5
            )

            image_buffer = io.BytesIO(
                image_bytes
            )

            pdf.setFont(
                "Helvetica-Bold",
                14
            )

            pdf.drawString(
                40,
                height - 35,
                turbine
            )

            pdf.drawImage(
                ImageReader(image_buffer),
                40,
                55,
                width=width - 80,
                height=height - 110,
                preserveAspectRatio=True,
                anchor="c"
            )

            pdf.showPage()

        except Exception:
            continue

    pdf.save()

    pdf_buffer.seek(0)

    return pdf_buffer


if KALEIDO_AVAILABLE:

    if st.button(
        "Generate PDF Report",
        use_container_width=True
    ):

        try:

            pdf_file = create_pdf_report(
                site,
                summary_df,
                results
            )

            if pdf_file is not None:

                st.download_button(
                    "Download PDF",
                    data=pdf_file,
                    file_name=(
                        f"{site}_Power_Curve_Report.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"PDF generation failed: {e}"
            )

else:

    st.info(
        "Kaleido is not installed. "
        "The dashboard graphs will still work, "
        "but PDF image export is unavailable."
    )
