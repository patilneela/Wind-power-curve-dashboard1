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


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Power Curve Analytics Report",
    layout="wide"
)


# ==========================================================
# LOGIN
# ==========================================================

def login_gate():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("Login Required")

    with st.form("login_form", clear_on_submit=False):

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        submitted = st.form_submit_button("Login")

    if submitted:

        cfg = st.secrets.get("auth", {})

        ok = (
            username == cfg.get("username")
            and
            password == cfg.get("password")
        )

        if ok:

            st.session_state.authenticated = True
            st.rerun()

        else:

            st.error("Invalid username or password")

    st.stop()


login_gate()


# ==========================================================
# LOGOUT
# ==========================================================

if st.sidebar.button("Logout", key="logout_btn"):

    st.session_state.authenticated = False
    st.rerun()


# ==========================================================
# KALEIDO CHECK
# ==========================================================

try:

    import kaleido

    KALEIDO_AVAILABLE = True

except Exception:

    KALEIDO_AVAILABLE = False


# ==========================================================
# PATH
# ==========================================================

BASE_DIR = os.path.dirname(__file__)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "Envision.png"
)

SITE_MASTER_XLSX = os.path.join(
    BASE_DIR,
    "site_master.xlsx"
)

SITE_MASTER_CSV = os.path.join(
    BASE_DIR,
    "site_master.csv"
)

BIN_SIZE = 0.5


# ==========================================================
# LOGO
# ==========================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    if os.path.exists(LOGO_PATH):

        st.image(
            LOGO_PATH,
            width=300
        )


# ==========================================================
# TITLE
# ==========================================================

st.title("Power Curve Analytics Report")

st.caption(
    "SCADA Data Driven Power Curve Analysis"
)


# ==========================================================
# HELPER - SITE MASTER PATH
# ==========================================================

def get_site_master_path():

    if os.path.exists(SITE_MASTER_XLSX):
        return SITE_MASTER_XLSX

    if os.path.exists(SITE_MASTER_CSV):
        return SITE_MASTER_CSV

    return None


# ==========================================================
# HELPER - PRESET DATE
# ==========================================================

def compute_preset_range(
    preset,
    today_ts
):

    today_date = today_ts.normalize().date()

    if preset == "Today":

        return today_date, today_date

    if preset == "This Week":

        monday = (
            today_date
            -
            timedelta(
                days=today_ts.weekday()
            )
        )

        return monday, today_date

    if preset == "Last Week":

        this_monday = (
            today_date
            -
            timedelta(
                days=today_ts.weekday()
            )
        )

        start = this_monday - timedelta(days=7)

        end = this_monday - timedelta(days=1)

        return start, end

    if preset == "This Month":

        start = today_date.replace(day=1)

        return start, today_date

    if preset == "Last Month":

        first_this_month = today_date.replace(day=1)

        last_month_end = (
            first_this_month
            -
            timedelta(days=1)
        )

        start = last_month_end.replace(day=1)

        return start, last_month_end

    return None


# ==========================================================
# COLUMN DETECTION
# ==========================================================

def find_column(
    columns,
    possible_names
):

    normalized = {}

    for col in columns:

        clean = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        normalized[clean] = col

    # Exact matching first

    for name in possible_names:

        key = (
            name
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if key in normalized:

            return normalized[key]

    # Partial matching

    for col in columns:

        low = str(col).strip().lower()

        for name in possible_names:

            if name.lower() in low:

                return col

    return None


# ==========================================================
# DETECT SCADA COLUMNS
# ==========================================================

def detect_scada_columns(df):

    columns = list(df.columns)

    turbine_col = find_column(
        columns,
        [
            "Name",
            "Turbine",
            "TurbineName",
            "WTG",
            "WTGName",
            "WindTurbine",
            "WindTurbineName"
        ]
    )

    site_col = find_column(
        columns,
        [
            "Site",
            "SiteName",
            "Plant",
            "PlantName",
            "Farm",
            "FarmName",
            "WindFarm",
            "WindFarmName",
            "Project",
            "ProjectName",
            "Location"
        ]
    )

    wind_col = find_column(
        columns,
        [
            "WindSpeedAve",
            "WindSpeed",
            "WindSpeedAverage",
            "WindSpeedAvg"
        ]
    )

    power_col = find_column(
        columns,
        [
            "ActivePWAve",
            "ActivePower",
            "ActivePowerAve",
            "Power",
            "PowerAve",
            "Active"
        ]
    )

    time_col = find_column(
        columns,
        [
            "Time",
            "Timestamp",
            "DateTime",
            "Date",
            "TimeStamp"
        ]
    )

    pitch_col = find_column(
        columns,
        [
            "BldPitch1Ave",
            "Pitch",
            "PitchAve",
            "BladePitch",
            "BladePitch1"
        ]
    )

    return (
        site_col,
        turbine_col,
        wind_col,
        power_col,
        time_col,
        pitch_col
    )


# ==========================================================
# LOAD SITE MASTER
# ==========================================================

@st.cache_data
def load_site_master():

    path = get_site_master_path()

    if path is None:

        return pd.DataFrame()

    try:

        if path.lower().endswith(".csv"):

            sm = pd.read_csv(path)

        else:

            sm = pd.read_excel(path)

        sm.columns = [
            str(c).strip()
            for c in sm.columns
        ]

        return sm

    except Exception:

        return pd.DataFrame()


site_master = load_site_master()


# ==========================================================
# SITE MASTER ADMIN
# ==========================================================

with st.sidebar.expander(
    "Site Master",
    expanded=False
):

    st.write(
        "Optional: Site Master can be used "
        "for turbine-to-site mapping and capacity."
    )

    sm_upload = st.file_uploader(
        "Upload Site Master",
        type=["xlsx", "csv"],
        key="site_master_upload"
    )

    if st.button(
        "Save Site Master",
        key="save_site_master"
    ):

        if sm_upload is None:

            st.warning(
                "Please select a Site Master file."
            )

        else:

            extension = os.path.splitext(
                sm_upload.name
            )[1].lower()

            target = os.path.join(
                BASE_DIR,
                f"site_master{extension}"
            )

            # Remove other format

            if extension == ".xlsx":

                if os.path.exists(
                    SITE_MASTER_CSV
                ):

                    os.remove(
                        SITE_MASTER_CSV
                    )

            if extension == ".csv":

                if os.path.exists(
                    SITE_MASTER_XLSX
                ):

                    os.remove(
                        SITE_MASTER_XLSX
                    )

            with open(
                target,
                "wb"
            ) as f:

                f.write(
                    sm_upload.getbuffer()
                )

            st.success(
                "Site Master saved successfully."
            )

            st.cache_data.clear()

            st.rerun()


# ==========================================================
# UPLOAD SCADA
# ==========================================================

st.sidebar.subheader(
    "Upload SCADA File"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload SCADA CSV",
    type=["csv"],
    key="scada_upload"
)


if uploaded_file is None:

    st.info(
        "Please upload the SCADA CSV file."
    )

    st.stop()


# ==========================================================
# LOAD SCADA
# ==========================================================

@st.cache_data(show_spinner=True)
def load_scada(file):

    chunksize = 200000

    chunks = []

    for chunk in pd.read_csv(
        file,
        chunksize=chunksize,
        low_memory=False,
        engine="c"
    ):

        chunks.append(chunk)

    df_local = pd.concat(
        chunks,
        ignore_index=True
    )

    df_local.columns = [
        str(c).strip()
        for c in df_local.columns
    ]

    return df_local


with st.spinner(
    "Loading SCADA data..."
):

    df = load_scada(
        uploaded_file
    )


if df.empty:

    st.error(
        "The uploaded SCADA file is empty."
    )

    st.stop()


# ==========================================================
# DETECT COLUMNS
# ==========================================================

(
    site_col,
    turbine_col,
    wind_col,
    power_col,
    time_col,
    pitch_col
) = detect_scada_columns(df)


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

missing = []

if site_col is None:

    missing.append(
        "Site"
    )

if turbine_col is None:

    missing.append(
        "Turbine / Name"
    )

if wind_col is None:

    missing.append(
        "Wind Speed"
    )

if power_col is None:

    missing.append(
        "Active Power"
    )

if time_col is None:

    missing.append(
        "Time / Timestamp"
    )


if missing:

    st.error(
        "Required columns could not be detected."
    )

    st.write(
        "Missing:"
    )

    for item in missing:

        st.write(
            f"- {item}"
        )

    st.write(
        "Columns detected in your CSV:"
    )

    st.code(
        "\n".join(
            map(str, df.columns)
        )
    )

    st.stop()


# ==========================================================
# NORMALIZE DATA
# ==========================================================

df[site_col] = (
    df[site_col]
    .astype(str)
    .str.strip()
)

df[turbine_col] = (
    df[turbine_col]
    .astype(str)
    .str.strip()
)

df[wind_col] = pd.to_numeric(
    df[wind_col],
    errors="coerce"
)

df[power_col] = pd.to_numeric(
    df[power_col],
    errors="coerce"
)

df[time_col] = pd.to_datetime(
    df[time_col],
    errors="coerce"
)

if pitch_col is not None:

    df[pitch_col] = pd.to_numeric(
        df[pitch_col],
        errors="coerce"
    )


# ==========================================================
# REMOVE INVALID DATA
# ==========================================================

required_for_cleaning = [
    site_col,
    turbine_col,
    wind_col,
    power_col,
    time_col
]

df = df.dropna(
    subset=required_for_cleaning
)


# ==========================================================
# REMOVE EMPTY SITE / TURBINE
# ==========================================================

df = df[
    (df[site_col].str.strip() != "")
    &
    (df[turbine_col].str.strip() != "")
]


if df.empty:

    st.error(
        "No valid SCADA rows are available."
    )

    st.stop()


# ==========================================================
# SITE LIST FROM ACTUAL SCADA DATA
# ==========================================================

sites = sorted(
    df[site_col]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


if not sites:

    st.error(
        "No sites were found in the uploaded SCADA data."
    )

    st.stop()


# ==========================================================
# SELECT SITE
# ==========================================================

selected_site = st.sidebar.selectbox(
    "Select Site",
    sites,
    key="selected_site"
)


# ==========================================================
# FILTER DATA BY SITE
# ==========================================================

site_df = df[
    df[site_col]
    .astype(str)
    .str.strip()
    ==
    selected_site
].copy()


if site_df.empty:

    st.warning(
        f"No SCADA data found for site: {selected_site}"
    )

    st.stop()


# ==========================================================
# DATE RANGE
# ==========================================================

st.sidebar.markdown(
    "### Date Range"
)

max_ts = site_df[time_col].max()

base_date = (
    max_ts.normalize().date()
    if pd.notna(max_ts)
    else pd.Timestamp.today().date()
)

DEFAULT_START = (
    base_date
    -
    timedelta(days=15)
)

DEFAULT_END = base_date


if "manual_start_date" not in st.session_state:

    st.session_state.manual_start_date = (
        DEFAULT_START
    )


if "manual_end_date" not in st.session_state:

    st.session_state.manual_end_date = (
        DEFAULT_END
    )


date_option = st.sidebar.selectbox(
    "Date Option",
    [
        "Clear",
        "Today",
        "This Week",
        "This Month",
        "Last Week",
        "Last Month",
        "Manual (Calendar)"
    ],
    key="date_option"
)


if date_option == "Manual (Calendar)":

    st.sidebar.markdown(
        "#### Manual Selection"
    )

    start_day = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.manual_start_date,
        key="manual_start_date"
    )

    end_day = st.sidebar.date_input(
        "End Date",
        value=st.session_state.manual_end_date,
        key="manual_end_date"
    )

elif date_option == "Clear":

    start_day = DEFAULT_START
    end_day = DEFAULT_END

else:

    rng = compute_preset_range(
        date_option,
        pd.Timestamp.today()
    )

    if rng is None:

        start_day = DEFAULT_START
        end_day = DEFAULT_END

    else:

        start_day, end_day = rng


# ==========================================================
# APPLY DATE FILTER
# ==========================================================

site_df["_date_only"] = (
    site_df[time_col]
    .dt.date
)


site_df = site_df[
    (site_df["_date_only"] >= start_day)
    &
    (site_df["_date_only"] <= end_day)
]


site_df = site_df.drop(
    columns=["_date_only"]
)


st.sidebar.caption(
    f"Applied Range: {start_day} → {end_day}"
)


if site_df.empty:

    st.warning(
        "No SCADA data is available for the selected site and date range."
    )

    st.stop()


# ==========================================================
# SITE INFORMATION
# ==========================================================

turbines = sorted(
    site_df[turbine_col]
    .dropna()
    .unique()
    .tolist()
)

num_turbines = len(turbines)


# ==========================================================
# CAPACITY
# ==========================================================

capacity_per_turbine = None

if not site_master.empty:

    sm_columns = list(
        site_master.columns
    )

    sm_site_col = find_column(
        sm_columns,
        [
            "Site",
            "SiteName",
            "Plant",
            "Project"
        ]
    )

    sm_capacity_col = find_column(
        sm_columns,
        [
            "Capacity_MW",
            "CapacityMW",
            "TurbineCapacityMW",
            "MW",
            "Capacity"
        ]
    )

    if (
        sm_site_col is not None
        and
        sm_capacity_col is not None
    ):

        temp_sm = site_master.copy()

        temp_sm[sm_site_col] = (
            temp_sm[sm_site_col]
            .astype(str)
            .str.strip()
        )

        matched = temp_sm[
            temp_sm[sm_site_col]
            ==
            selected_site
        ]

        if not matched.empty:

            capacity_per_turbine = pd.to_numeric(
                matched.iloc[0][sm_capacity_col],
                errors="coerce"
            )

            if pd.isna(
                capacity_per_turbine
            ):

                capacity_per_turbine = None


if capacity_per_turbine is None:

    capacity_per_turbine = 3.3


total_capacity = (
    num_turbines
    *
    float(capacity_per_turbine)
)


# ==========================================================
# HEADER
# ==========================================================

st.subheader(
    f"{selected_site} | "
    f"{num_turbines} Turbines | "
    f"{capacity_per_turbine} MW Each | "
    f"Total: {round(total_capacity, 2)} MW"
)

st.markdown(
    f"**Date Range:** {start_day} → {end_day}"
)


# ==========================================================
# DATA INFORMATION
# ==========================================================

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        "Site",
        selected_site
    )


with info2:

    st.metric(
        "Turbines",
        num_turbines
    )


with info3:

    st.metric(
        "SCADA Records",
        f"{len(site_df):,}"
    )


with info4:

    st.metric(
        "Capacity",
        f"{round(total_capacity, 2)} MW"
    )


# ==========================================================
# VIEW MODE
# ==========================================================

mode = st.sidebar.radio(
    "Select View",
    [
        "Single Turbine",
        "Compare Turbines",
        "Show All Turbines"
    ],
    key="mode_radio"
)


if mode == "Single Turbine":

    turbines_to_show = [
        st.sidebar.selectbox(
            "Select Turbine",
            turbines,
            key="single_turbine"
        )
    ]


elif mode == "Compare Turbines":

    turbines_to_show = st.sidebar.multiselect(
        "Select Turbines",
        turbines,
        key="compare_turbines"
    )


else:

    turbines_to_show = turbines


if not turbines_to_show:

    st.info(
        "Please select at least one turbine."
    )

    st.stop()


# ==========================================================
# PROCESS ACTUAL SCADA POWER CURVE
# ==========================================================

def process_turbine(
    turbine_name
):

    df_t = site_df[
        site_df[turbine_col]
        ==
        turbine_name
    ].copy()


    # ------------------------------------------------------
    # BASIC FILTERING
    # ------------------------------------------------------

    df_t = df_t[
        (df_t[wind_col] >= 3)
        &
        (df_t[wind_col] <= 25)
        &
        (df_t[power_col] >= 0)
    ]


    # ------------------------------------------------------
    # PITCH FILTER IF AVAILABLE
    # ------------------------------------------------------

    if pitch_col is not None:

        df_t = df_t[
            (df_t[pitch_col] >= -5)
            &
            (df_t[pitch_col] <= 5)
        ]


    if len(df_t) < 10:

        return None


    # ------------------------------------------------------
    # CREATE WIND BINS
    # ------------------------------------------------------

    df_t["WindBin"] = (
        np.floor(
            df_t[wind_col] / BIN_SIZE
        )
        *
        BIN_SIZE
    ).round(2)


    # ------------------------------------------------------
    # ACTUAL POWER CURVE
    # ------------------------------------------------------

    actual = (
        df_t
        .groupby("WindBin")
        .agg(
            AvgPower=(
                power_col,
                "mean"
            ),

            MinPower=(
                power_col,
                "min"
            ),

            MaxPower=(
                power_col,
                "max"
            ),

            Count=(
                power_col,
                "count"
            ),

            AvgWind=(
                wind_col,
                "mean"
            )
        )
        .reset_index()
    )


    actual = actual.sort_values(
        "WindBin"
    )


    # ------------------------------------------------------
    # SMOOTH CURVE
    # ------------------------------------------------------

    actual["SmoothPower"] = (
        actual["AvgPower"]
    )


    valid_points = (
        actual["AvgPower"]
        .notna()
    )


    if valid_points.sum() >= 7:

        try:

            actual.loc[
                valid_points,
                "SmoothPower"
            ] = savgol_filter(
                actual.loc[
                    valid_points,
                    "AvgPower"
                ].values,
                7,
                2
            )

        except Exception:

            actual["SmoothPower"] = (
                actual["AvgPower"]
            )


    # ------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------

    average_power = (
        df_t[power_col]
        .mean()
    )

    maximum_power = (
        df_t[power_col]
        .max()
    )

    minimum_power = (
        df_t[power_col]
        .min()
    )

    std_power = (
        df_t[power_col]
        .std()
    )

    average_wind = (
        df_t[wind_col]
        .mean()
    )


    return {
        "raw": df_t,
        "curve": actual,
        "average_power": average_power,
        "maximum_power": maximum_power,
        "minimum_power": minimum_power,
        "std_power": std_power,
        "average_wind": average_wind
    }


# ==========================================================
# POWER CURVE GRAPH
# ==========================================================

def plot_power_curve(
    result,
    turbine_name
):

    df_t = result["raw"]

    curve = result["curve"]


    fig = go.Figure()


    # ------------------------------------------------------
    # RAW SCADA POINTS
    # ------------------------------------------------------

    fig.add_trace(
        go.Scattergl(
            x=df_t[wind_col],
            y=df_t[power_col],
            mode="markers",

            marker=dict(
                size=4,
                opacity=0.25
            ),

            name="SCADA Data",

            hovertemplate=
            "Wind: %{x:.2f} m/s"
            "<br>"
            "Power: %{y:.2f}"
            "<extra></extra>"
        )
    )


    # ------------------------------------------------------
    # ACTUAL BIN AVERAGE
    # ------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=curve["WindBin"],
            y=curve["AvgPower"],
            mode="lines+markers",

            line=dict(
                width=3
            ),

            marker=dict(
                size=6
            ),

            name="Actual Power Curve",

            hovertemplate=
            "Wind Bin: %{x:.2f} m/s"
            "<br>"
            "Average Power: %{y:.2f}"
            "<extra></extra>"
        )
    )


    # ------------------------------------------------------
    # SMOOTH CURVE
    # ------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=curve["WindBin"],
            y=curve["SmoothPower"],
            mode="lines",

            line=dict(
                width=4,
                dash="solid"
            ),

            name="Smoothed Power Curve",

            hovertemplate=
            "Wind: %{x:.2f} m/s"
            "<br>"
            "Power: %{y:.2f}"
            "<extra></extra>"
        )
    )


    # ------------------------------------------------------
    # GRAPH LAYOUT
    # ------------------------------------------------------

    fig.update_layout(

        title=dict(
            text=f"{turbine_name} - Actual SCADA Power Curve",
            font=dict(
                size=20
            )
        ),

        xaxis=dict(

            title="Wind Speed (m/s)",

            showgrid=True,

            zeroline=False,

            rangeslider=dict(
                visible=False
            )
        ),

        yaxis=dict(

            title="Active Power",

            showgrid=True,

            zeroline=False
        ),

        height=550,

        hovermode="closest",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),

        margin=dict(
            l=60,
            r=30,
            t=90,
            b=60
        )
    )


    return fig


# ==========================================================
# ANALYSIS COMMENT
# ==========================================================

def generate_comment(
    result
):

    if result is None:

        return "Insufficient SCADA data."


    avg_power = result[
        "average_power"
    ]

    max_power = result[
        "maximum_power"
    ]

    avg_wind = result[
        "average_wind"
    ]

    return (
        f"Average Wind Speed: "
        f"{avg_wind:.2f} m/s\n\n"

        f"Average Power: "
        f"{avg_power:.2f}\n\n"

        f"Maximum Power: "
        f"{max_power:.2f}\n\n"

        f"Valid SCADA Records: "
        f"{len(result['raw']):,}"
    )


# ==========================================================
# PROCESS ALL SELECTED TURBINES
# ==========================================================

results = []

figures = []


for turbine in turbines_to_show:

    result = process_turbine(
        turbine
    )


    if result is None:

        continue


    curve = result["curve"]


    # ------------------------------------------------------
    # CURVE QUALITY
    # ------------------------------------------------------

    number_of_bins = len(
        curve
    )


    max_power = result[
        "maximum_power"
    ]


    avg_power = result[
        "average_power"
    ]


    std_power = result[
        "std_power"
    ]


    results.append({

        "Turbine": turbine,

        "SCADA Records":
            len(result["raw"]),

        "Wind Bins":
            number_of_bins,

        "Average Wind":
            round(
                result["average_wind"],
                2
            ),

        "Average Power":
            round(
                avg_power,
                2
            ),

        "Maximum Power":
            round(
                max_power,
                2
            ),

        "Power Std Dev":
            round(
                std_power,
                2
            )
    })


    fig = plot_power_curve(
        result,
        turbine
    )


    comment = generate_comment(
        result
    )


    figures.append(
        (
            turbine,
            fig,
            comment
        )
    )


# ==========================================================
# DISPLAY POWER CURVES
# ==========================================================

st.divider()

st.subheader(
    "Actual SCADA Power Curves"
)


if not figures:

    st.warning(
        "No valid power curves could be generated for the selected turbines."
    )

    st.stop()


# ==========================================================
# DISPLAY TWO GRAPHS PER ROW
# ==========================================================

cols = st.columns(2)


for i, (
    turbine,
    fig,
    comment
) in enumerate(figures):

    with cols[i % 2]:

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander(
            f"Analysis - {turbine}",
            expanded=False
        ):

            st.text(
                comment
            )


# ==========================================================
# TURBINE DATA TABLE
# ==========================================================

st.divider()

st.subheader(
    "Turbine Performance Summary"
)


results_df = pd.DataFrame(
    results
)


if not results_df.empty:

    # Sort by average power

    results_df = (
        results_df
        .sort_values(
            by="Average Power",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    # Add rank

    results_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(results_df) + 1
        )
    )


    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# TURBINE COMPARISON
# ==========================================================

if (
    mode == "Compare Turbines"
    and
    len(figures) > 1
):

    st.divider()

    st.subheader(
        "Turbine Comparison"
    )


    comparison_fig = go.Figure()


    for turbine, fig, comment in figures:

        result = process_turbine(
            turbine
        )

        if result is None:
            continue

        curve = result["curve"]


        comparison_fig.add_trace(
            go.Scatter(
                x=curve["WindBin"],
                y=curve["SmoothPower"],
                mode="lines+markers",
                name=turbine,

                hovertemplate=
                f"{turbine}"
                "<br>"
                "Wind: %{x:.2f} m/s"
                "<br>"
                "Power: %{y:.2f}"
                "<extra></extra>"
            )
        )


    comparison_fig.update_layout(

        title="Turbine Actual Power Curve Comparison",

        xaxis_title="Wind Speed (m/s)",

        yaxis_title="Active Power",

        height=650,

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )


    st.plotly_chart(
        comparison_fig,
        use_container_width=True
    )


# ==========================================================
# SITE-WIDE POWER CURVE
# ==========================================================

st.divider()

st.subheader(
    "Selected Site - Overall Power Curve"
)


site_plot_df = site_df.copy()


site_plot_df = site_plot_df[
    (site_plot_df[wind_col] >= 3)
    &
    (site_plot_df[wind_col] <= 25)
    &
    (site_plot_df[power_col] >= 0)
]


if not site_plot_df.empty:

    site_plot_df["WindBin"] = (
        np.floor(
            site_plot_df[wind_col]
            /
            BIN_SIZE
        )
        *
        BIN_SIZE
    ).round(2)


    site_curve = (
        site_plot_df
        .groupby("WindBin")
        .agg(
            AveragePower=(
                power_col,
                "mean"
            ),

            MinimumPower=(
                power_col,
                "min"
            ),

            MaximumPower=(
                power_col,
                "max"
            ),

            Records=(
                power_col,
                "count"
            )
        )
        .reset_index()
    )


    site_curve = site_curve.sort_values(
        "WindBin"
    )


    if len(site_curve) >= 7:

        try:

            site_curve["SmoothPower"] = (
                savgol_filter(
                    site_curve[
                        "AveragePower"
                    ],
                    7,
                    2
                )
            )

        except Exception:

            site_curve["SmoothPower"] = (
                site_curve[
                    "AveragePower"
                ]
            )

    else:

        site_curve["SmoothPower"] = (
            site_curve[
                "AveragePower"
            ]
        )


    site_fig = go.Figure()


    site_fig.add_trace(
        go.Scatter(
            x=site_curve["WindBin"],
            y=site_curve["AveragePower"],
            mode="lines+markers",

            line=dict(
                width=3
            ),

            marker=dict(
                size=6
            ),

            name="Site Average"
        )
    )


    site_fig.add_trace(
        go.Scatter(
            x=site_curve["WindBin"],
            y=site_curve["SmoothPower"],
            mode="lines",

            line=dict(
                width=4
            ),

            name="Site Smoothed Curve"
        )
    )


    site_fig.update_layout(

        title=f"{selected_site} - Overall Actual Power Curve",

        xaxis_title="Wind Speed (m/s)",

        yaxis_title="Average Active Power",

        height=600,

        hovermode="x unified"
    )


    st.plotly_chart(
        site_fig,
        use_container_width=True
    )


# ==========================================================
# DOWNLOAD CURVE DATA
# ==========================================================

st.divider()

st.subheader(
    "Download Power Curve Data"
)


download_frames = []


for turbine in turbines_to_show:

    result = process_turbine(
        turbine
    )

    if result is None:
        continue


    curve = result[
        "curve"
    ].copy()


    curve.insert(
        0,
        "Turbine",
        turbine
    )


    curve.insert(
        0,
        "Site",
        selected_site
    )


    download_frames.append(
        curve
    )


if download_frames:

    combined_curve_data = pd.concat(
        download_frames,
        ignore_index=True
    )


    csv_data = (
        combined_curve_data
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(

        label="Download Power Curve Data CSV",

        data=csv_data,

        file_name=(
            f"{selected_site}"
            "_Power_Curve_Data.csv"
        ),

        mime="text/csv"
    )


# ==========================================================
# PDF REPORT
# ==========================================================

st.divider()

st.subheader(
    "PDF Report"
)


try:

    pdf_buffer = io.BytesIO()


    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=landscape(A4)
    )


    width, height = landscape(A4)


    # ------------------------------------------------------
    # PDF HEADER
    # ------------------------------------------------------

    if os.path.exists(
        LOGO_PATH
    ):

        try:

            pdf.drawImage(
                LOGO_PATH,
                30,
                height - 80,
                width=120,
                height=40,
                preserveAspectRatio=True
            )

        except Exception:

            pass


    pdf.setFont(
        "Helvetica-Bold",
        16
    )


    pdf.drawString(
        170,
        height - 40,
        "Power Curve Analytics Report"
    )


    pdf.setFont(
        "Helvetica",
        10
    )


    pdf.drawString(
        170,
        height - 60,
        f"Site: {selected_site}"
    )


    pdf.drawString(
        170,
        height - 75,
        f"Date Range: {start_day} to {end_day}"
    )


    pdf.drawString(
        170,
        height - 90,
        f"Turbines: {num_turbines}"
    )


    # ------------------------------------------------------
    # GRAPH PAGES
    # ------------------------------------------------------

    y = height - 120


    for turbine, fig, comment in figures:

        if KALEIDO_AVAILABLE:

            try:

                img = fig.to_image(
                    format="png"
                )


                img_reader = ImageReader(
                    io.BytesIO(img)
                )


                if y < 270:

                    pdf.showPage()

                    y = height - 60


                pdf.drawImage(
                    img_reader,
                    30,
                    y - 220,
                    width=500,
                    height=210,
                    preserveAspectRatio=True
                )


                pdf.setFont(
                    "Helvetica-Bold",
                    11
                )


                pdf.drawString(
                    550,
                    y - 40,
                    turbine
                )


                pdf.setFont(
                    "Helvetica",
                    9
                )


                comment_lines = (
                    comment.split("\n")
                )


                text_y = y - 60


                for line in comment_lines:

                    pdf.drawString(
                        550,
                        text_y,
                        line
                    )

                    text_y -= 14


                y -= 240


            except Exception:

                pass


    # ------------------------------------------------------
    # SUMMARY PAGE
    # ------------------------------------------------------

    pdf.showPage()


    pdf.setFont(
        "Helvetica-Bold",
        14
    )


    pdf.drawString(
        30,
        height - 40,
        "Turbine Performance Summary"
    )


    y = height - 80


    pdf.setFont(
        "Helvetica",
        9
    )


    if not results_df.empty:

        for _, row in results_df.iterrows():

            line = (
                f"Rank {row['Rank']} | "
                f"{row['Turbine']} | "
                f"Records: {row['SCADA Records']} | "
                f"Avg Wind: {row['Average Wind']} | "
                f"Avg Power: {row['Average Power']} | "
                f"Max Power: {row['Maximum Power']}"
            )


            pdf.drawString(
                30,
                y,
                line
            )


            y -= 18


            if y < 40:

                pdf.showPage()

                y = height - 40


    pdf.save()


    pdf_buffer.seek(0)


    st.download_button(

        label="Download Full Dashboard Report (PDF)",

        data=pdf_buffer.getvalue(),

        file_name=(
            f"{selected_site}"
            "_Power_Curve_Report.pdf"
        ),

        mime="application/pdf"
    )


except Exception as e:

    st.warning(
        "PDF generation failed."
    )

    st.code(
        str(e)
    )


# ==========================================================
# DEBUG / DATA INFORMATION
# ==========================================================

with st.expander(
    "Detected SCADA Columns"
):

    st.write(
        {
            "Site Column": site_col,
            "Turbine Column": turbine_col,
            "Wind Speed Column": wind_col,
            "Power Column": power_col,
            "Time Column": time_col,
            "Pitch Column": pitch_col
        }
    )


with st.expander(
    "Selected Site Data Preview"
):

    st.dataframe(
        site_df.head(100),
        use_container_width=True
    )
