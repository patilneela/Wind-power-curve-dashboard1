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

if st.sidebar.button(
    "Logout",
    key="logout_btn"
):

    st.session_state.authenticated = False
    st.rerun()


# ==========================================================
# SAFE KALEIDO CHECK
# ==========================================================

try:

    import kaleido

    KALEIDO_AVAILABLE = True

except Exception:

    KALEIDO_AVAILABLE = False


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(__file__)

NEW_REFERENCE_PATH = os.path.join(
    BASE_DIR,
    "reference.xlsx"
)

SITE_MASTER_XLSX = os.path.join(
    BASE_DIR,
    "site_master.xlsx"
)

SITE_MASTER_CSV = os.path.join(
    BASE_DIR,
    "site_master.csv"
)

LOGO_PATH = os.path.join(
    BASE_DIR,
    "Envision.png"
)

BIN_SIZE = 0.5


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
# DATE PRESET
# ==========================================================

def compute_preset_range(
    preset: str,
    today_ts: pd.Timestamp
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

        start = (
            this_monday
            -
            timedelta(days=7)
        )

        end = (
            this_monday
            -
            timedelta(days=1)
        )

        return start, end

    if preset == "This Month":

        start = today_date.replace(day=1)

        return start, today_date

    if preset == "Last Month":

        first_this_month = (
            today_date.replace(day=1)
        )

        last_month_end = (
            first_this_month
            -
            timedelta(days=1)
        )

        start = last_month_end.replace(
            day=1
        )

        return start, last_month_end

    return None


# ==========================================================
# COLUMN DETECTION HELPER
# ==========================================================

def find_column(
    columns,
    possible_names
):

    # ------------------------------------------------------
    # Exact normalized match
    # ------------------------------------------------------

    normalized = {}

    for col in columns:

        key = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        normalized[key] = col


    for name in possible_names:

        key = (
            name
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if key in normalized:

            return normalized[key]


    # ------------------------------------------------------
    # Partial match
    # ------------------------------------------------------

    for col in columns:

        col_low = (
            str(col)
            .strip()
            .lower()
        )

        for name in possible_names:

            if (
                name.lower()
                in col_low
            ):

                return col

    return None


# ==========================================================
# DETECT REFERENCE COLUMNS
# ==========================================================

def detect_reference_columns(
    reference_df
):

    columns = list(
        reference_df.columns
    )


    reference_site_col = find_column(
        columns,
        [
            "Site",
            "SiteName",
            "Site Name",
            "Plant",
            "PlantName",
            "Farm",
            "FarmName",
            "Project",
            "ProjectName",
            "WindFarm",
            "WindFarmName",
            "Location"
        ]
    )


    reference_turbine_col = find_column(
        columns,
        [
            "Turbine",
            "TurbineName",
            "Turbine Name",
            "Name",
            "WTG",
            "WTGName",
            "WTG Name",
            "WindTurbine",
            "WindTurbineName",
            "Machine",
            "MachineName"
        ]
    )


    return (
        reference_site_col,
        reference_turbine_col
    )


# ==========================================================
# LOAD NEW REFERENCE FILE
# ==========================================================

@st.cache_data
def load_new_reference():

    if not os.path.exists(
        NEW_REFERENCE_PATH
    ):

        return None, None, None


    try:

        ref = pd.read_excel(
            NEW_REFERENCE_PATH
        )

        ref.columns = [
            str(c).strip()
            for c in ref.columns
        ]


        site_col, turbine_col = (
            detect_reference_columns(
                ref
            )
        )


        if (
            site_col is None
            or
            turbine_col is None
        ):

            return (
                ref,
                None,
                None
            )


        ref[site_col] = (
            ref[site_col]
            .astype(str)
            .str.strip()
        )


        ref[turbine_col] = (
            ref[turbine_col]
            .astype(str)
            .str.strip()
        )


        ref = ref[
            (ref[site_col] != "")
            &
            (ref[turbine_col] != "")
        ].copy()


        return (
            ref,
            site_col,
            turbine_col
        )


    except Exception as e:

        st.error(
            "Unable to read the new reference Excel."
        )

        st.code(
            str(e)
        )

        return None, None, None


# ==========================================================
# DEFAULT SITE CAPACITY
# ==========================================================

DEFAULT_SITE_CAPACITY = {

    site: 3.3

    for site in [

        "CIP Hatalageri",
        "JSW Tuljapur",
        "Blupine Sagapara",
        "Kalavad GJ",
        "Kalavad_PH2",
        "AMP_Energy",
        "Wanki",
        "CleanMax Motadevaliya",
        "Ayana Amerli",
        "Mahadev PH1",
        "Blupine-I, Ambada-GJ",
        "ACME Shapar",
        "FP_Kudligi",
        "Sprng TN",
        "Otha Pithalpur-GJ",
        "AMGEPL,Kurnool AP",
        "ReNew1_Gadag",
        "partner Ottapidaum",
        "Cleanmax SANATHALI",
        "Cleanmax Babra",
        "RenfraEnergy Trichy",
        "RENEW-03 Sholapur",
        "Renew2 Chandwad",
        "ReNew-4 Patoda",
        "Clean max Jagalur",
        "Sembcorp Tuticorin",
        "Renew-4 Kudligi",
        "Renew Otha",
        "Cleanmax Honavad",
        "Blueleaf Agar",
        "JSW_Sandur",
        "India_Hero_Doni"
    ]
}


# ==========================================================
# LOAD SITE CAPACITY
# ==========================================================

@st.cache_data
def load_site_capacity():

    capacity = dict(
        DEFAULT_SITE_CAPACITY
    )

    path = get_site_master_path()

    if path is None:

        return capacity


    try:

        if path.lower().endswith(
            ".csv"
        ):

            sm = pd.read_csv(path)

        else:

            sm = pd.read_excel(path)


        sm.columns = [
            str(c).strip()
            for c in sm.columns
        ]


        site_col = find_column(
            sm.columns,
            [
                "Site",
                "SiteName",
                "Site Name",
                "Plant",
                "Project"
            ]
        )


        cap_col = find_column(
            sm.columns,
            [
                "Capacity_MW",
                "CapacityMW",
                "Capacity",
                "TurbineCapacityMW",
                "MW"
            ]
        )


        if (
            site_col is None
            or
            cap_col is None
        ):

            return capacity


        sm[site_col] = (
            sm[site_col]
            .astype(str)
            .str.strip()
        )


        sm[cap_col] = pd.to_numeric(
            sm[cap_col],
            errors="coerce"
        )


        sm = sm.dropna(
            subset=[
                site_col,
                cap_col
            ]
        )


        for _, row in sm.iterrows():

            capacity[
                row[site_col]
            ] = float(
                row[cap_col]
            )


        return capacity


    except Exception:

        return capacity


SITE_CAPACITY = load_site_capacity()


# ==========================================================
# TABS
# ==========================================================

tab_dashboard, tab_admin = st.tabs(
    [
        "Dashboard",
        "Site Add-on"
    ]
)


# ==========================================================
# ADMIN TAB
# ==========================================================

with tab_admin:

    st.subheader(
        "Site Add-on"
    )

    st.divider()


    # ======================================================
    # NEW REFERENCE FILE
    # ======================================================

    st.markdown(
        "## 1) New Reference Excel"
    )


    if os.path.exists(
        NEW_REFERENCE_PATH
    ):

        st.success(
            "Reference Excel found: "
            f"`{os.path.basename(NEW_REFERENCE_PATH)}`"
        )

    else:

        st.warning(
            "No reference Excel found. "
            "Upload the new reference Excel."
        )


    reference_upload = st.file_uploader(

        "Upload / Replace New Reference Excel",

        type=["xlsx"],

        key="new_reference_upload"
    )


    c1, c2 = st.columns(2)


    with c1:

        if st.button(
            "Save / Replace Reference Excel",
            type="primary",
            key="save_reference"
        ):

            if reference_upload is None:

                st.error(
                    "Please select the new reference Excel first."
                )

            else:

                with open(
                    NEW_REFERENCE_PATH,
                    "wb"
                ) as f:

                    f.write(
                        reference_upload.getbuffer()
                    )


                st.success(
                    "New reference Excel saved successfully."
                )


                st.cache_data.clear()

                st.rerun()


    with c2:

        if st.button(
            "Delete Reference Excel",
            key="delete_reference"
        ):

            if os.path.exists(
                NEW_REFERENCE_PATH
            ):

                os.remove(
                    NEW_REFERENCE_PATH
                )

                st.success(
                    "Reference Excel deleted."
                )

                st.cache_data.clear()

                st.rerun()

            else:

                st.info(
                    "No reference file found."
                )


    # ======================================================
    # REFERENCE PREVIEW
    # ======================================================

    if os.path.exists(
        NEW_REFERENCE_PATH
    ):

        with st.expander(
            "Preview New Reference Excel"
        ):

            try:

                preview = pd.read_excel(
                    NEW_REFERENCE_PATH
                )

                st.dataframe(
                    preview.head(100),
                    use_container_width=True
                )


                ref_site_col, ref_turbine_col = (
                    detect_reference_columns(
                        preview
                    )
                )


                st.write(
                    "**Detected Site Column:**",
                    ref_site_col
                )


                st.write(
                    "**Detected Turbine Column:**",
                    ref_turbine_col
                )


            except Exception as e:

                st.error(
                    "Unable to preview reference Excel."
                )

                st.code(
                    str(e)
                )


    st.divider()


    # ======================================================
    # SITE MASTER
    # ======================================================

    st.markdown(
        "## 2) Site Master"
    )


    existing_sm = (
        get_site_master_path()
    )


    if existing_sm:

        st.success(
            f"Site Master found: "
            f"`{os.path.basename(existing_sm)}`"
        )

    else:

        st.info(
            "Site Master is optional."
        )


    sm_upload = st.file_uploader(

        "Upload / Replace Site Master",

        type=["xlsx", "csv"],

        key="sm_upload"
    )


    c3, c4 = st.columns(2)


    with c3:

        if st.button(
            "Save / Replace Site Master",
            type="primary",
            key="save_site_master"
        ):

            if sm_upload is None:

                st.error(
                    "Please select a Site Master file."
                )

            else:

                ext = os.path.splitext(
                    sm_upload.name
                )[1].lower()


                target = os.path.join(
                    BASE_DIR,
                    f"site_master{ext}"
                )


                if (
                    ext == ".xlsx"
                    and
                    os.path.exists(
                        SITE_MASTER_CSV
                    )
                ):

                    os.remove(
                        SITE_MASTER_CSV
                    )


                if (
                    ext == ".csv"
                    and
                    os.path.exists(
                        SITE_MASTER_XLSX
                    )
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
                    "Site Master saved."
                )


                st.cache_data.clear()

                st.rerun()


    with c4:

        if st.button(
            "Delete Site Master",
            key="delete_site_master"
        ):

            deleted = False


            if os.path.exists(
                SITE_MASTER_XLSX
            ):

                os.remove(
                    SITE_MASTER_XLSX
                )

                deleted = True


            if os.path.exists(
                SITE_MASTER_CSV
            ):

                os.remove(
                    SITE_MASTER_CSV
                )

                deleted = True


            if deleted:

                st.success(
                    "Site Master deleted."
                )

                st.cache_data.clear()

                st.rerun()

            else:

                st.info(
                    "No Site Master found."
                )


# ==========================================================
# DASHBOARD TAB
# ==========================================================

with tab_dashboard:


    # ======================================================
    # UPLOAD SCADA
    # ======================================================

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
            "Please upload SCADA CSV."
        )

        st.stop()


    # ======================================================
    # LOAD NEW REFERENCE
    # ======================================================

    (
        reference_df,
        reference_site_col,
        reference_turbine_col
    ) = load_new_reference()


    if reference_df is None:

        st.error(
            "New Reference Excel is missing."
        )

        st.info(
            "Go to the 'Site Add-on' tab and "
            "upload the new reference Excel."
        )

        st.stop()


    if (
        reference_site_col is None
        or
        reference_turbine_col is None
    ):

        st.error(
            "The new reference Excel must contain "
            "a Site column and a Turbine/Name/WTG column."
        )

        st.write(
            "Columns detected in reference file:"
        )

        st.code(
            "\n".join(
                map(
                    str,
                    reference_df.columns
                )
            )
        )

        st.stop()


    # ======================================================
    # LOAD SCADA
    # ======================================================

    @st.cache_data(show_spinner=True)
    def load_scada(file):

        chunksize = 200000

        chunks = pd.read_csv(
            file,
            chunksize=chunksize,
            low_memory=False,
            engine="c"
        )


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
        "Loading SCADA file..."
    ):

        df = load_scada(
            uploaded_file
        )


    if df.empty:

        st.error(
            "SCADA file is empty."
        )

        st.stop()


    # ======================================================
    # DETECT SCADA COLUMNS
    # ======================================================

    scada_site_col = find_column(
        df.columns,
        [
            "Site",
            "SiteName",
            "Site Name",
            "Plant",
            "PlantName",
            "Farm",
            "FarmName",
            "Project",
            "ProjectName",
            "WindFarm",
            "WindFarmName"
        ]
    )


    scada_turbine_col = find_column(
        df.columns,
        [
            "Name",
            "Turbine",
            "TurbineName",
            "Turbine Name",
            "WTG",
            "WTGName",
            "WTG Name",
            "WindTurbine",
            "WindTurbineName"
        ]
    )


    wind_col = find_column(
        df.columns,
        [
            "WindSpeedAve",
            "WindSpeed",
            "WindSpeedAverage",
            "WindSpeedAvg"
        ]
    )


    power_col = find_column(
        df.columns,
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
        df.columns,
        [
            "Time",
            "Timestamp",
            "DateTime",
            "Date",
            "TimeStamp"
        ]
    )


    pitch_col = find_column(
        df.columns,
        [
            "BldPitch1Ave",
            "Pitch",
            "PitchAve",
            "BladePitch",
            "BladePitch1"
        ]
    )


    # ======================================================
    # SCADA COLUMN CHECK
    # ======================================================

    missing = []


    if scada_turbine_col is None:

        missing.append(
            "Turbine / Name / WTG"
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
            "Required SCADA columns were not detected."
        )


        st.write(
            "Missing:"
        )


        for x in missing:

            st.write(
                f"- {x}"
            )


        st.write(
            "SCADA columns detected:"
        )


        st.code(
            "\n".join(
                map(
                    str,
                    df.columns
                )
            )
        )


        st.stop()


    # ======================================================
    # NORMALIZE SCADA
    # ======================================================

    df[scada_turbine_col] = (
        df[scada_turbine_col]
        .astype(str)
        .str.strip()
    )


    if scada_site_col is not None:

        df[scada_site_col] = (
            df[scada_site_col]
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


    df = df.dropna(
        subset=[
            scada_turbine_col,
            wind_col,
            power_col,
            time_col
        ]
    )


    if df.empty:

        st.error(
            "No valid SCADA data remains after cleaning."
        )

        st.stop()


    # ======================================================
    # GET SITES FROM NEW REFERENCE
    # ======================================================

    reference_df[reference_site_col] = (
        reference_df[
            reference_site_col
        ]
        .astype(str)
        .str.strip()
    )


    reference_df[reference_turbine_col] = (
        reference_df[
            reference_turbine_col
        ]
        .astype(str)
        .str.strip()
    )


    reference_sites = sorted(
        reference_df[
            reference_site_col
        ]
        .dropna()
        .unique()
        .tolist()
    )


    if not reference_sites:

        st.error(
            "No sites found in the new reference file."
        )

        st.stop()


    # ======================================================
    # SELECT SITE
    # ======================================================

    site = st.sidebar.selectbox(
        "Select Site",
        reference_sites,
        key="site_select"
    )


    # ======================================================
    # GET TURBINES FROM NEW REFERENCE
    # ======================================================

    site_reference = reference_df[
        reference_df[
            reference_site_col
        ]
        ==
        site
    ].copy()


    reference_turbines = sorted(
        site_reference[
            reference_turbine_col
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    # ======================================================
    # FILTER SCADA USING NEW REFERENCE TURBINES
    # ======================================================

    if scada_site_col is not None:

        site_scada = df[
            df[scada_site_col]
            ==
            site
        ].copy()

    else:

        site_scada = df.copy()


    # ------------------------------------------------------
    # Match turbines from reference
    # ------------------------------------------------------

    site_scada = site_scada[
        site_scada[
            scada_turbine_col
        ]
        .isin(
            reference_turbines
        )
    ].copy()


    # ======================================================
    # HANDLE TURBINE NAME FORMAT DIFFERENCES
    # ======================================================

    if site_scada.empty:

        # Try case-insensitive matching

        reference_map = {
            str(x).strip().lower(): x
            for x in reference_turbines
        }


        scada_names = (
            site_scada[
                scada_turbine_col
            ]
            if not site_scada.empty
            else pd.Series(
                df[
                    scada_turbine_col
                ]
                .dropna()
                .unique()
            )
        )


        # If direct site filtering caused empty result,
        # retry without SCADA site column.

        if scada_site_col is not None:

            temp_scada = df.copy()

            temp_scada[
                scada_turbine_col
            ] = (
                temp_scada[
                    scada_turbine_col
                ]
                .astype(str)
                .str.strip()
            )


            temp_scada[
                "_turbine_lower"
            ] = (
                temp_scada[
                    scada_turbine_col
                ]
                .str.lower()
            )


            allowed_lower = set(
                reference_map.keys()
            )


            temp_scada = temp_scada[
                temp_scada[
                    "_turbine_lower"
                ]
                .isin(
                    allowed_lower
                )
            ].copy()


            if not temp_scada.empty:

                site_scada = temp_scada.drop(
                    columns=[
                        "_turbine_lower"
                    ]
                )


    # ======================================================
    # IF STILL EMPTY
    # ======================================================

    if site_scada.empty:

        st.error(
            f"No SCADA data was found for site '{site}' "
            "using the turbines listed in the new reference file."
        )


        st.write(
            "Turbines found in new reference:"
        )

        st.write(
            reference_turbines
        )


        if scada_site_col is not None:

            st.write(
                "SCADA sites detected:"
            )

            st.write(
                sorted(
                    df[
                        scada_site_col
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )


        st.stop()


    # ======================================================
    # DATE FILTER
    # ======================================================

    st.sidebar.markdown(
        "### Date Range"
    )


    max_ts = site_scada[
        time_col
    ].max()


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


    if (
        "manual_start_date"
        not in st.session_state
    ):

        st.session_state.manual_start_date = (
            DEFAULT_START
        )


    if (
        "manual_end_date"
        not in st.session_state
    ):

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


    # ======================================================
    # APPLY DATE FILTER
    # ======================================================

    site_scada["_date_only"] = (
        site_scada[
            time_col
        ].dt.date
    )


    site_scada = site_scada[
        (
            site_scada[
                "_date_only"
            ]
            >=
            start_day
        )
        &
        (
            site_scada[
                "_date_only"
            ]
            <=
            end_day
        )
    ].copy()


    site_scada = site_scada.drop(
        columns=[
            "_date_only"
        ]
    )


    st.sidebar.caption(
        f"Applied Range: "
        f"{start_day} → {end_day}"
    )


    if site_scada.empty:

        st.warning(
            "No SCADA data available for "
            "the selected date range."
        )

        st.stop()


    # ======================================================
    # FINAL TURBINE LIST
    # ======================================================

    turbines = sorted(
        site_scada[
            scada_turbine_col
        ]
        .dropna()
        .unique()
        .tolist()
    )


    num_turbines = len(
        turbines
    )


    # ======================================================
    # CAPACITY
    # ======================================================

    capacity_per_turbine = (
        SITE_CAPACITY.get(
            site,
            3.3
        )
    )


    total_capacity = (
        num_turbines
        *
        capacity_per_turbine
    )


    # ======================================================
    # HEADER
    # ======================================================

    st.subheader(

        f"{site} | "
        f"{num_turbines} Turbines | "
        f"{capacity_per_turbine} MW Each | "
        f"Total: {round(total_capacity, 2)} MW"

    )


    st.markdown(
        f"**Date Range:** "
        f"{start_day} → {end_day}"
    )


    # ======================================================
    # SUMMARY METRICS
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Selected Site",
            site
        )


    with c2:

        st.metric(
            "Reference Turbines",
            len(reference_turbines)
        )


    with c3:

        st.metric(
            "SCADA Turbines",
            num_turbines
        )


    with c4:

        st.metric(
            "SCADA Records",
            f"{len(site_scada):,}"
        )


    # ======================================================
    # REFERENCE TURBINE CHECK
    # ======================================================

    missing_from_scada = sorted(
        list(
            set(reference_turbines)
            -
            set(turbines)
        )
    )


    extra_scada_turbines = sorted(
        list(
            set(turbines)
            -
            set(reference_turbines)
        )
    )


    with st.expander(
        "Reference Turbine Filtering Details"
    ):

        st.write(
            f"**Turbines in new reference:** "
            f"{len(reference_turbines)}"
        )


        st.write(
            f"**Turbines available in filtered SCADA:** "
            f"{len(turbines)}"
        )


        if missing_from_scada:

            st.warning(
                "Turbines present in reference "
                "but not available in selected SCADA/date range:"
            )

            st.write(
                missing_from_scada
            )


        if extra_scada_turbines:

            st.info(
                "SCADA turbines not listed in the "
                "new reference were excluded:"
            )

            st.write(
                extra_scada_turbines
            )


    # ======================================================
    # VIEW MODE
    # ======================================================

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

        selected_turbine = (
            st.sidebar.selectbox(
                "Select Turbine",
                turbines,
                key="single_turbine"
            )
        )


        turbines_to_show = [
            selected_turbine
        ]


    elif mode == "Compare Turbines":

        turbines_to_show = (
            st.sidebar.multiselect(
                "Select Turbines",
                turbines,
                key="compare_turbines"
            )
        )


    else:

        turbines_to_show = turbines


    if not turbines_to_show:

        st.info(
            "Please select at least one turbine."
        )

        st.stop()


    # ======================================================
    # PROCESS TURBINE
    # ======================================================

    def process_turbine(
        turbine
    ):

        df_t = site_scada[
            site_scada[
                scada_turbine_col
            ]
            ==
            turbine
        ].copy()


        # --------------------------------------------------
        # EXISTING FILTER LOGIC
        # --------------------------------------------------

        df_t = df_t[
            (
                df_t[
                    wind_col
                ]
                >=
                3
            )
            &
            (
                df_t[
                    wind_col
                ]
                <=
                25
            )
            &
            (
                df_t[
                    power_col
                ]
                > 0
            )
        ]


        # --------------------------------------------------
        # PITCH FILTER
        # --------------------------------------------------

        if pitch_col is not None:

            df_t = df_t[
                (
                    df_t[
                        pitch_col
                    ]
                    >=
                    -5
                )
                &
                (
                    df_t[
                        pitch_col
                    ]
                    <=
                    5
                )
            ]


        # --------------------------------------------------
        # MINIMUM DATA
        # --------------------------------------------------

        if len(df_t) < 30:

            return None


        # --------------------------------------------------
        # STANDARD DEVIATION
        # --------------------------------------------------

        std_dev = (
            df_t[
                power_col
            ].std()
        )


        # --------------------------------------------------
        # WIND BIN
        # --------------------------------------------------

        df_t["WindBin"] = (

            np.floor(

                df_t[
                    wind_col
                ]
                /
                BIN_SIZE

            )
            *
            BIN_SIZE

        ).round(6)


        # --------------------------------------------------
        # ACTUAL POWER CURVE
        # --------------------------------------------------

        actual = (

            df_t

            .groupby(
                "WindBin"
            )

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
                )

            )

            .reset_index()

        )


        actual = actual.sort_values(
            "WindBin"
        )


        # --------------------------------------------------
        # SMOOTHED ACTUAL CURVE
        # --------------------------------------------------

        actual[
            "SmoothPower"
        ] = actual[
            "AvgPower"
        ]


        if len(actual) >= 7:

            try:

                actual[
                    "SmoothPower"
                ] = savgol_filter(

                    actual[
                        "AvgPower"
                    ].values,

                    7,

                    2

                )

            except Exception:

                actual[
                    "SmoothPower"
                ] = actual[
                    "AvgPower"
                ]


        # --------------------------------------------------
        # ADD POWER DEVIATION FROM TURBINE MEAN
        # --------------------------------------------------
        #
        # This is NOT reference-curve deviation.
        # It is only used for basic statistical analysis.
        #

        avg_power = (
            df_t[
                power_col
            ].mean()
        )


        max_power = (
            df_t[
                power_col
            ].max()
        )


        min_power = (
            df_t[
                power_col
            ].min()
        )


        avg_wind = (
            df_t[
                wind_col
            ].mean()
        )


        return (

            df_t,
            actual,
            avg_power,
            max_power,
            min_power,
            avg_wind,
            std_dev

        )


    # ======================================================
    # PLOT GRAPH
    # ======================================================

    def plot_graph(
        result,
        title
    ):

        (
            df_t,
            actual,
            avg_power,
            max_power,
            min_power,
            avg_wind,
            std_dev
        ) = result


        fig = go.Figure()


        # --------------------------------------------------
        # RAW SCATTER
        # --------------------------------------------------

        fig.add_trace(

            go.Scattergl(

                x=df_t[
                    wind_col
                ],

                y=df_t[
                    power_col
                ],

                mode="markers",

                marker=dict(

                    size=4,

                    opacity=0.30

                ),

                name="SCADA Points",

                hovertemplate=(

                    "Wind Speed: "
                    "%{x:.2f} m/s"

                    "<br>"

                    "Power: "
                    "%{y:.2f}"

                    "<extra></extra>"

                )

            )

        )


        # --------------------------------------------------
        # ACTUAL AVERAGE POWER CURVE
        # --------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=actual[
                    "WindBin"
                ],

                y=actual[
                    "AvgPower"
                ],

                mode="lines+markers",

                line=dict(
                    width=3
                ),

                marker=dict(
                    size=6
                ),

                name="Actual Power Curve",

                hovertemplate=(

                    "Wind Bin: "
                    "%{x:.2f} m/s"

                    "<br>"

                    "Average Power: "
                    "%{y:.2f}"

                    "<br>"

                    "<extra></extra>"

                )

            )

        )


        # --------------------------------------------------
        # SMOOTH CURVE
        # --------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=actual[
                    "WindBin"
                ],

                y=actual[
                    "SmoothPower"
                ],

                mode="lines",

                line=dict(

                    width=4

                ),

                name="Smoothed Curve",

                hovertemplate=(

                    "Wind Speed: "
                    "%{x:.2f} m/s"

                    "<br>"

                    "Smoothed Power: "
                    "%{y:.2f}"

                    "<extra></extra>"

                )

            )

        )


        # --------------------------------------------------
        # LAYOUT
        # --------------------------------------------------

        fig.update_layout(

            title=dict(

                text=(
                    f"{title} - "
                    "Actual SCADA Power Curve"
                ),

                font=dict(
                    size=20
                )

            ),

            xaxis=dict(

                title="Wind Speed (m/s)",

                showgrid=True,

                zeroline=False

            ),

            yaxis=dict(

                title="Active Power",

                showgrid=True,

                zeroline=False

            ),

            height=550,

            hovermode="x unified",

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


    # ======================================================
    # ANALYSIS
    # ======================================================

    def generate_comment(
        result
    ):

        if result is None:

            return (
                "Insufficient SCADA data."
            )


        (
            df_t,
            actual,
            avg_power,
            max_power,
            min_power,
            avg_wind,
            std_dev
        ) = result


        return (

            f"Average Wind Speed: "
            f"{avg_wind:.2f} m/s\n\n"

            f"Average Power: "
            f"{avg_power:.2f}\n\n"

            f"Maximum Power: "
            f"{max_power:.2f}\n\n"

            f"Minimum Power: "
            f"{min_power:.2f}\n\n"

            f"Power Std Dev: "
            f"{std_dev:.2f}\n\n"

            f"Valid SCADA Records: "
            f"{len(df_t):,}\n\n"

            f"Wind Bins: "
            f"{len(actual)}"

        )


    # ======================================================
    # DISPLAY
    # ======================================================

    st.divider()

    st.subheader(
        "Power Curve Analysis"
    )


    cols = st.columns(2)

    results = []

    figures = []


    for i, turbine in enumerate(
        turbines_to_show
    ):

        result = process_turbine(
            turbine
        )


        if result is None:

            continue


        fig = plot_graph(
            result,
            turbine
        )


        comment = generate_comment(
            result
        )


        with cols[
            i % 2
        ]:

            st.plotly_chart(
                fig,
                use_container_width=True
            )


            st.markdown(
                "### Analysis"
            )


            st.code(
                comment
            )


        figures.append(
            (
                turbine,
                fig,
                comment
            )
        )


        (
            df_t,
            actual,
            avg_power,
            max_power,
            min_power,
            avg_wind,
            std_dev
        ) = result


        results.append({

            "Turbine":
                turbine,

            "SCADA Records":
                len(df_t),

            "Wind Bins":
                len(actual),

            "Average Wind":
                round(
                    avg_wind,
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

            "Minimum Power":
                round(
                    min_power,
                    2
                ),

            "Power Std Dev":
                round(
                    std_dev,
                    2
                )

        })


    # ======================================================
    # RANKING
    # ======================================================

    st.divider()

    st.subheader(
        "Turbine Ranking"
    )


    results_df = pd.DataFrame(
        results
    )


    if not results_df.empty:

        # --------------------------------------------------
        # Rank according to average power
        # --------------------------------------------------

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


    # ======================================================
    # COMPARISON GRAPH
    # ======================================================

    if (
        mode == "Compare Turbines"
        and
        len(figures) > 1
    ):

        st.divider()

        st.subheader(
            "Turbine Power Curve Comparison"
        )


        comparison_fig = go.Figure()


        for turbine in turbines_to_show:

            result = process_turbine(
                turbine
            )


            if result is None:

                continue


            (
                df_t,
                actual,
                avg_power,
                max_power,
                min_power,
                avg_wind,
                std_dev
            ) = result


            comparison_fig.add_trace(

                go.Scatter(

                    x=actual[
                        "WindBin"
                    ],

                    y=actual[
                        "SmoothPower"
                    ],

                    mode="lines+markers",

                    name=turbine,

                    hovertemplate=(

                        f"{turbine}"

                        "<br>"

                        "Wind: "
                        "%{x:.2f} m/s"

                        "<br>"

                        "Power: "
                        "%{y:.2f}"

                        "<extra></extra>"

                    )

                )

            )


        comparison_fig.update_layout(

            title=(
                f"{site} - "
                "Actual Power Curve Comparison"
            ),

            xaxis_title=(
                "Wind Speed (m/s)"
            ),

            yaxis_title=(
                "Active Power"
            ),

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


    # ======================================================
    # SITE-WIDE POWER CURVE
    # ======================================================

    st.divider()

    st.subheader(
        f"{site} - Overall Actual Power Curve"
    )


    site_plot_df = site_scada.copy()


    site_plot_df = site_plot_df[

        (
            site_plot_df[
                wind_col
            ]
            >=
            3
        )

        &

        (
            site_plot_df[
                wind_col
            ]
            <=
            25
        )

        &

        (
            site_plot_df[
                power_col
            ]
            >=
            0
        )

    ]


    if (
        pitch_col is not None
    ):

        site_plot_df = site_plot_df[

            (
                site_plot_df[
                    pitch_col
                ]
                >=
                -5
            )

            &

            (
                site_plot_df[
                    pitch_col
                ]
                <=
                5
            )

        ]


    if not site_plot_df.empty:

        site_plot_df[
            "WindBin"
        ] = (

            np.floor(

                site_plot_df[
                    wind_col
                ]
                /
                BIN_SIZE

            )
            *
            BIN_SIZE

        ).round(6)


        site_curve = (

            site_plot_df

            .groupby(
                "WindBin"
            )

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


        site_curve = (
            site_curve
            .sort_values(
                "WindBin"
            )
        )


        site_curve[
            "SmoothPower"
        ] = site_curve[
            "AveragePower"
        ]


        if len(site_curve) >= 7:

            try:

                site_curve[
                    "SmoothPower"
                ] = savgol_filter(

                    site_curve[
                        "AveragePower"
                    ].values,

                    7,

                    2

                )

            except Exception:

                pass


        site_fig = go.Figure()


        site_fig.add_trace(

            go.Scatter(

                x=site_curve[
                    "WindBin"
                ],

                y=site_curve[
                    "AveragePower"
                ],

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

                x=site_curve[
                    "WindBin"
                ],

                y=site_curve[
                    "SmoothPower"
                ],

                mode="lines",

                line=dict(
                    width=4
                ),

                name="Site Smoothed Curve"

            )

        )


        site_fig.update_layout(

            title=(
                f"{site} - "
                "Overall Actual Power Curve"
            ),

            xaxis_title=(
                "Wind Speed (m/s)"
            ),

            yaxis_title=(
                "Average Active Power"
            ),

            height=600,

            hovermode="x unified"

        )


        st.plotly_chart(
            site_fig,
            use_container_width=True
        )


    # ======================================================
    # DOWNLOAD POWER CURVE DATA
    # ======================================================

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


        (
            df_t,
            actual,
            avg_power,
            max_power,
            min_power,
            avg_wind,
            std_dev
        ) = result


        curve = actual.copy()


        curve.insert(
            0,
            "Turbine",
            turbine
        )


        curve.insert(
            0,
            "Site",
            site
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

            label=(
                "Download Power Curve Data CSV"
            ),

            data=csv_data,

            file_name=(
                f"{site}"
                "_Power_Curve_Data.csv"
            ),

            mime="text/csv"

        )


    # ======================================================
    # PDF REPORT
    # ======================================================

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


        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

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

            f"Site: {site}"

        )


        pdf.drawString(

            170,

            height - 75,

            f"Date Range: "
            f"{start_day} to {end_day}"

        )


        pdf.drawString(

            170,

            height - 90,

            f"Turbines: {num_turbines}"

        )


        pdf.drawString(

            170,

            height - 105,

            "Power Curve Source: "
            "Actual SCADA Data"

        )


        # --------------------------------------------------
        # GRAPHS
        # --------------------------------------------------

        y = height - 130


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


                    lines = comment.split(
                        "\n"
                    )


                    text_y = y - 60


                    for line in lines:

                        pdf.drawString(

                            550,

                            text_y,

                            line

                        )

                        text_y -= 14


                    y -= 240


                except Exception:

                    pass


        # --------------------------------------------------
        # SUMMARY PAGE
        # --------------------------------------------------

        pdf.showPage()


        pdf.setFont(
            "Helvetica-Bold",
            14
        )


        pdf.drawString(

            30,

            height - 40,

            "Turbine Ranking Summary"

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

                    f"Records: "
                    f"{row['SCADA Records']} | "

                    f"Avg Wind: "
                    f"{row['Average Wind']} | "

                    f"Avg Power: "
                    f"{row['Average Power']} | "

                    f"Max Power: "
                    f"{row['Maximum Power']}"

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

            label=(
                "Download Full Dashboard Report (PDF)"
            ),

            data=pdf_buffer.getvalue(),

            file_name=(
                f"{site}"
                "_Power_Curve_Report.pdf"
            ),

            mime="application/pdf"

        )


    except Exception as e:

        st.error(
            "PDF generation failed."
        )

        st.code(
            str(e)
        )


    # ======================================================
    # DEBUG INFORMATION
    # ======================================================

    with st.expander(
        "Detected SCADA Columns"
    ):

        st.write({

            "SCADA Site Column":
                scada_site_col,

            "SCADA Turbine Column":
                scada_turbine_col,

            "Wind Speed Column":
                wind_col,

            "Power Column":
                power_col,

            "Time Column":
                time_col,

            "Pitch Column":
                pitch_col,

            "Reference Site Column":
                reference_site_col,

            "Reference Turbine Column":
                reference_turbine_col

        })


    with st.expander(
        "Selected Site SCADA Data Preview"
    ):

        st.dataframe(

            site_scada.head(100),

            use_container_width=True

        )
