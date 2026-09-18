import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os
import io
import re

from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Power Curve Analytics Report",
    layout="wide"
)


# =========================================================
# SIMPLE LOGIN
# =========================================================

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
            and password == cfg.get("password")
        )

        if ok:

            st.session_state.authenticated = True
            st.rerun()

        else:

            st.error("Invalid username or password")

    st.stop()


login_gate()


# =========================================================
# LOGOUT
# =========================================================

if st.sidebar.button(
    "Logout",
    key="logout_btn"
):

    st.session_state.authenticated = False
    st.rerun()


# =========================================================
# KALEIDO CHECK
# =========================================================

try:

    import kaleido  # noqa: F401

    KALEIDO_AVAILABLE = True

except Exception:

    KALEIDO_AVAILABLE = False


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# CHANGE ONLY THIS FILE NAME IF REQUIRED
# =========================================================

REFERENCE_FILE_NAME = (
    "All Sites Specific Power Curve data_ENV.xlsx"
)

REF_FILE_PATH = os.path.join(
    BASE_DIR,
    REFERENCE_FILE_NAME
)

SITE_MASTER_XLSX = os.path.join(
    BASE_DIR,
    "site_master.xlsx"
)

SITE_MASTER_CSV = os.path.join(
    BASE_DIR,
    "site_master.csv"
)


# =========================================================
# CONSTANTS
# =========================================================

BIN_SIZE = 0.5

MAX_REFERENCE_WIND = 30

REFERENCE_START_WIND = 3

REFERENCE_END_WIND = 25


# =========================================================
# DEFAULT SITE CAPACITY
# =========================================================

DEFAULT_SITE_CAPACITY = {

    "CleanMax -Gujarat (*Den-1.142)": 3.3,

    "Renew-4 -Kudligi- KA (*Den-1.076)": 3.3,

    "Renew-4-Otha - GJ (*Den-1.153)": 3.3,

    "Renew -Pithalur-GJ  (*Den-1.153)": 3.3,

    "CleanMax-Jagalur-KA (*Den-1.070)": 3.3,

    "Fourthpartner-Ottapidaram-TN  (*Den-1.145)": 3.3,

    "Sembcorp Tuticorin-TN  (*Den-1.145)": 3.3,

    "AMGPEL Kurnool": 3.3,

    "JSW Sandur KA": 3.3,

    "Fourthpartner Kudligi KA": 3.3,

    "Ayana Amreli GJ": 3.3,

    "Sprng Mulanur TN (1.125Kg/m3)": 3.3,

    "ACME Shapur GJ (1.136Kg/m3)": 3.3,

    "Cleanmax Honavad KA (1.105Kg/m3)": 3.3,

    "Renfra trichy TN": 3.3,

    "NSL_AP (1.089 Kg/m3)": 3.3
}


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value):

    if value is None:
        return ""

    try:

        if pd.isna(value):
            return ""

    except Exception:

        pass

    text = str(value)

    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")

    # Replace line breaks/tabs
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Normalize multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


def site_names_match(
    requested_site,
    reference_site
):

    a = normalize_text(
        requested_site
    )

    b = normalize_text(
        reference_site
    )

    if not a or not b:
        return False

    # Exact normalized match
    if a == b:
        return True

    # Remove common formatting differences
    a2 = re.sub(
        r"[\s\-_]+",
        "",
        a
    )

    b2 = re.sub(
        r"[\s\-_]+",
        "",
        b
    )

    if a2 == b2:
        return True

    # One contains the other
    if a2 in b2 or b2 in a2:
        return True

    return False


# =========================================================
# SITE MASTER PATH
# =========================================================

def get_site_master_path():

    if os.path.exists(
        SITE_MASTER_XLSX
    ):

        return SITE_MASTER_XLSX

    if os.path.exists(
        SITE_MASTER_CSV
    ):

        return SITE_MASTER_CSV

    return None


# =========================================================
# DATE PRESET
# =========================================================

def compute_preset_range(
    preset: str,
    today_ts: pd.Timestamp
):

    today_date = (
        today_ts
        .normalize()
        .date()
    )

    if preset == "Today":

        return (
            today_date,
            today_date
        )

    if preset == "This Week":

        monday = (
            today_date
            - timedelta(
                days=today_ts.weekday()
            )
        )

        return (
            monday,
            today_date
        )

    if preset == "Last Week":

        this_monday = (
            today_date
            - timedelta(
                days=today_ts.weekday()
            )
        )

        start = (
            this_monday
            - timedelta(days=7)
        )

        end = (
            this_monday
            - timedelta(days=1)
        )

        return (
            start,
            end
        )

    if preset == "This Month":

        start = today_date.replace(
            day=1
        )

        return (
            start,
            today_date
        )

    if preset == "Last Month":

        first_this_month = (
            today_date.replace(day=1)
        )

        last_month_end = (
            first_this_month
            - timedelta(days=1)
        )

        start = last_month_end.replace(
            day=1
        )

        return (
            start,
            last_month_end
        )

    return None


# =========================================================
# LOGO
# =========================================================

logo_path = os.path.join(
    BASE_DIR,
    "Envision.png"
)

col1, col2, col3 = st.columns(
    [1, 2, 1]
)

with col2:

    if os.path.exists(
        logo_path
    ):

        st.image(
            logo_path,
            width=300
        )


# =========================================================
# TITLE
# =========================================================

st.title(
    "Power Curve Analytics Report"
)


# =========================================================
# SITE CAPACITY
# =========================================================

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

            sm = pd.read_csv(
                path
            )

        else:

            sm = pd.read_excel(
                path
            )

        sm.columns = [
            str(c).strip()
            for c in sm.columns
        ]

        site_col = None
        cap_col = None

        for c in sm.columns:

            c_lower = (
                str(c)
                .strip()
                .lower()
            )

            if c_lower in [
                "site",
                "site_name",
                "sitename",
                "plant",
                "project"
            ]:

                site_col = c

            if c_lower in [
                "capacity_mw",
                "capacity",
                "turbine_capacity_mw",
                "mw"
            ]:

                cap_col = c

        if (
            site_col is None
            or cap_col is None
        ):

            st.warning(
                "Site Master columns not recognized. "
                "Required: Site + Capacity_MW."
            )

            return capacity

        sm = sm[
            [
                site_col,
                cap_col
            ]
        ].dropna()

        sm[site_col] = (
            sm[site_col]
            .astype(str)
            .str.strip()
        )

        sm[cap_col] = pd.to_numeric(
            sm[cap_col],
            errors="coerce"
        )

        sm = sm.dropna()

        for _, row in sm.iterrows():

            capacity[
                row[site_col]
            ] = float(
                row[cap_col]
            )

        return capacity

    except Exception as e:

        st.warning(
            "Failed to read Site Master. "
            "Using default site list."
        )

        st.code(
            str(e)
        )

        return capacity


SITE_CAPACITY = load_site_capacity()


# =========================================================
# TABS
# =========================================================

tab_dashboard, tab_admin = st.tabs(
    [
        "Dashboard",
        "Site Add-on"
    ]
)


# =========================================================
# ADMIN TAB
# =========================================================

with tab_admin:

    st.subheader(
        "Site Add-on"
    )

    st.divider()

    # =====================================================
    # REFERENCE FILE
    # =====================================================

    st.markdown(
        "## 1) Reference Excel"
    )

    if os.path.exists(
        REF_FILE_PATH
    ):

        st.success(
            "Reference file found: "
            f"`{os.path.basename(REF_FILE_PATH)}`"
        )

    else:

        st.warning(
            "Reference file missing."
        )

    ref_upload = st.file_uploader(
        "Upload / Replace Reference Excel (.xlsx)",
        type=["xlsx"],
        key="ref_upload"
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            "Save / Replace Reference Excel",
            type="primary"
        ):

            if ref_upload is None:

                st.error(
                    "Please choose an .xlsx file first."
                )

            else:

                with open(
                    REF_FILE_PATH,
                    "wb"
                ) as f:

                    f.write(
                        ref_upload.getbuffer()
                    )

                st.success(
                    "New reference Excel saved. "
                    "Old reference has been replaced."
                )

                st.cache_data.clear()

                st.rerun()

    with c2:

        if st.button(
            "Delete Reference Excel"
        ):

            if os.path.exists(
                REF_FILE_PATH
            ):

                os.remove(
                    REF_FILE_PATH
                )

                st.success(
                    "Reference Excel deleted."
                )

                st.cache_data.clear()

                st.rerun()

            else:

                st.info(
                    "No reference file to delete."
                )


    # =====================================================
    # REFERENCE PREVIEW
    # =====================================================

    if os.path.exists(
        REF_FILE_PATH
    ):

        with st.expander(
            "Preview Reference Excel"
        ):

            try:

                preview = pd.read_excel(
                    REF_FILE_PATH,
                    header=None
                )

                st.dataframe(
                    preview.head(50),
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "Unable to read reference Excel."
                )

                st.code(
                    str(e)
                )


    st.divider()


    # =====================================================
    # SITE MASTER
    # =====================================================

    st.markdown(
        "## 2) Site Master"
    )

    existing_sm = (
        get_site_master_path()
    )

    if existing_sm:

        st.success(
            "Site Master found: "
            f"`{os.path.basename(existing_sm)}`"
        )

    else:

        st.info(
            "No Site Master file found. "
            "Default site list will be used."
        )


    sm_upload = st.file_uploader(
        "Upload / Replace Site Master (.xlsx or .csv)",
        type=["xlsx", "csv"],
        key="sm_upload"
    )


    c3, c4 = st.columns(2)


    with c3:

        if st.button(
            "Save / Replace Site Master",
            type="primary"
        ):

            if sm_upload is None:

                st.error(
                    "Please choose a file first."
                )

            else:

                ext = os.path.splitext(
                    sm_upload.name
                )[1].lower()

                if ext not in [
                    ".xlsx",
                    ".csv"
                ]:

                    st.error(
                        "Only .xlsx or .csv supported."
                    )

                else:

                    target = os.path.join(
                        BASE_DIR,
                        f"site_master{ext}"
                    )

                    if (
                        target != SITE_MASTER_XLSX
                        and os.path.exists(
                            SITE_MASTER_XLSX
                        )
                    ):

                        os.remove(
                            SITE_MASTER_XLSX
                        )

                    if (
                        target != SITE_MASTER_CSV
                        and os.path.exists(
                            SITE_MASTER_CSV
                        )
                    ):

                        os.remove(
                            SITE_MASTER_CSV
                        )

                    with open(
                        target,
                        "wb"
                    ) as f:

                        f.write(
                            sm_upload.getbuffer()
                        )

                    st.success(
                        "Site Master replaced."
                    )

                    st.cache_data.clear()

                    st.rerun()


    with c4:

        if st.button(
            "Delete Site Master"
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
                    "No Site Master to delete."
                )


    existing_sm = (
        get_site_master_path()
    )

    if existing_sm:

        with st.expander(
            "Preview Site Master"
        ):

            try:

                if existing_sm.endswith(
                    ".csv"
                ):

                    sm_df = pd.read_csv(
                        existing_sm
                    )

                else:

                    sm_df = pd.read_excel(
                        existing_sm
                    )

                st.dataframe(
                    sm_df.head(50),
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "Unable to read Site Master."
                )

                st.code(
                    str(e)
                )


# =========================================================
# DASHBOARD
# =========================================================

with tab_dashboard:

    # =====================================================
    # SCADA UPLOAD
    # =====================================================

    st.sidebar.subheader(
        "Upload SCADA File"
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload SCADA CSV",
        type=["csv"],
        key="scada_upload"
    )

    if uploaded_file is None:

        st.warning(
            "Please upload SCADA file."
        )

        st.stop()


    # =====================================================
    # REFERENCE FILE CHECK
    # =====================================================

    if not os.path.exists(
        REF_FILE_PATH
    ):

        st.error(
            "Reference Excel is missing. "
            "Upload it in the Site Add-on tab."
        )

        st.stop()


    # =====================================================
    # SITE + MODE
    # =====================================================

    site = st.sidebar.selectbox(
        "Select Site",
        list(
            SITE_CAPACITY.keys()
        ),
        key="site_select"
    )

    mode = st.sidebar.radio(
        "Select View",
        [
            "Single Turbine",
            "Compare Turbines",
            "Show All Turbines"
        ],
        key="mode_radio"
    )


    # =====================================================
    # LOAD SCADA
    # =====================================================

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

        df_local.columns = (
            df_local.columns
            .astype(str)
            .str.strip()
        )


        # -------------------------------------------------
        # TURBINE NAME
        # -------------------------------------------------

        if "Name" not in df_local.columns:

            raise ValueError(
                "SCADA CSV must contain "
                "'Name' column."
            )


        # -------------------------------------------------
        # WIND COLUMN
        # -------------------------------------------------

        wind_candidates = [
            c
            for c in df_local.columns
            if (
                "wind" in c.lower()
                and (
                    "speed" in c.lower()
                    or "ws" in c.lower()
                    or "ave" in c.lower()
                    or "avg" in c.lower()
                )
            )
        ]

        if not wind_candidates:

            wind_candidates = [
                c
                for c in df_local.columns
                if "wind" in c.lower()
            ]

        if not wind_candidates:

            raise ValueError(
                "Wind speed column not found."
            )

        wind_col = wind_candidates[0]


        # -------------------------------------------------
        # POWER COLUMN
        # -------------------------------------------------

        power_candidates = [
            c
            for c in df_local.columns
            if (
                "power" in c.lower()
                or "active" in c.lower()
            )
        ]

        if not power_candidates:

            raise ValueError(
                "Power column not found."
            )

        power_col = power_candidates[0]


        # -------------------------------------------------
        # TIME COLUMN
        # -------------------------------------------------

        time_candidates = [
            c
            for c in df_local.columns
            if (
                "time" in c.lower()
                or "date" in c.lower()
            )
        ]

        if not time_candidates:

            raise ValueError(
                "Time/date column not found."
            )

        time_col = time_candidates[0]


        # -------------------------------------------------
        # PITCH COLUMN
        # -------------------------------------------------

        pitch_candidates = [
            c
            for c in df_local.columns
            if "pitch" in c.lower()
        ]

        if pitch_candidates:

            pitch_col = pitch_candidates[0]

            df_local[pitch_col] = pd.to_numeric(
                df_local[pitch_col],
                errors="coerce"
            )

        else:

            pitch_col = "_Pitch_Default"

            df_local[pitch_col] = 0.0


        # -------------------------------------------------
        # CONVERT DATA TYPES
        # -------------------------------------------------

        df_local[time_col] = pd.to_datetime(
            df_local[time_col],
            errors="coerce"
        )

        df_local[wind_col] = pd.to_numeric(
            df_local[wind_col],
            errors="coerce"
        )

        df_local[power_col] = pd.to_numeric(
            df_local[power_col],
            errors="coerce"
        )


        # -------------------------------------------------
        # CLEAN
        # -------------------------------------------------

        df_local = df_local.dropna(
            subset=[
                wind_col,
                power_col,
                time_col
            ]
        )

        df_local["Name"] = (
            df_local["Name"]
            .astype(str)
            .str.strip()
        )

        return (
            df_local,
            wind_col,
            power_col,
            time_col,
            pitch_col
        )


    with st.spinner(
        "Loading SCADA file..."
    ):

        try:

            (
                df,
                wind_col,
                power_col,
                time_col,
                pitch_col
            ) = load_scada(
                uploaded_file
            )

        except Exception as e:

            st.error(
                "Unable to load SCADA file."
            )

            st.code(
                str(e)
            )

            st.stop()


    if df.empty:

        st.warning(
            "SCADA file has no valid rows."
        )

        st.stop()


    # =====================================================
    # DATE FILTER
    # =====================================================

    st.sidebar.markdown(
        "### Date Range"
    )

    max_ts = df[time_col].max()

    base_date = (
        max_ts.normalize().date()
        if pd.notna(max_ts)
        else pd.Timestamp.today().date()
    )


    DEFAULT_START = (
        base_date
        - timedelta(days=15)
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

        st.session_state.manual_start_date = (
            DEFAULT_START
        )

        st.session_state.manual_end_date = (
            DEFAULT_END
        )

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


    # =====================================================
    # APPLY DATE FILTER
    # =====================================================

    df["_date_only"] = (
        df[time_col].dt.date
    )

    df = df[
        (df["_date_only"] >= start_day)
        &
        (df["_date_only"] <= end_day)
    ]

    df = df.drop(
        columns=["_date_only"]
    )


    st.sidebar.caption(
        f"Applied Range: "
        f"{start_day} → {end_day}"
    )


    if df.empty:

        st.warning(
            "No SCADA data available "
            "for selected date range."
        )

        st.stop()


    # =====================================================
    # HEADER
    # =====================================================

    num_turbines = (
        df["Name"].nunique()
    )

    capacity_per_turbine = (
        SITE_CAPACITY.get(
            site,
            3.3
        )
    )

    total_capacity = (
        num_turbines
        * capacity_per_turbine
    )


    st.subheader(
        f"{site} | "
        f"{num_turbines} Turbines | "
        f"{capacity_per_turbine} MW Each | "
        f"Total: {round(total_capacity, 2)} MW"
    )

    st.markdown(
        f"Date Range: "
        f"{start_day} → {end_day}"
    )


    # =====================================================
    # NEW REFERENCE LOADER
    # =====================================================

    @st.cache_data
    def load_reference(site_name):

        """
        NEW REFERENCE EXCEL FORMAT

        Example:

        Row 1:
        Wind Speed [m/s] | Power [kW] | Power [kW] | ...

        Row 2:
        blank              | Site A     | Site B     | ...

        Row 3 onward:
        3.0                | 7          | 9          | ...
        3.5                | 86         | 66         | ...
        4.0                | 222        | 191        | ...
        ...

        This function finds the column corresponding
        to the selected site and uses:

            Column A = Wind Speed
            Selected site column = Power
        """

        # -------------------------------------------------
        # OPEN EXCEL
        # -------------------------------------------------

        try:

            excel_file = pd.ExcelFile(
                REF_FILE_PATH
            )

        except Exception as e:

            raise ValueError(
                "Unable to open reference Excel: "
                f"{e}"
            )


        site_candidates = []


        # =================================================
        # CHECK EVERY SHEET
        # =================================================

        for sheet in excel_file.sheet_names:

            try:

                raw = pd.read_excel(
                    REF_FILE_PATH,
                    sheet_name=sheet,
                    header=None
                )

            except Exception:

                continue


            if raw.empty:

                continue


            # -------------------------------------------------
            # REMOVE COMPLETELY EMPTY ROWS/COLUMNS
            # -------------------------------------------------

            raw = raw.dropna(
                axis=0,
                how="all"
            )

            raw = raw.dropna(
                axis=1,
                how="all"
            )

            if raw.empty:

                continue


            # -------------------------------------------------
            # SEARCH FIRST FEW ROWS FOR SITE NAMES
            # -------------------------------------------------

            max_header_rows = min(
                10,
                len(raw)
            )


            for col_index in range(
                1,
                len(raw.columns)
            ):

                possible_site_names = []


                for row_index in range(
                    max_header_rows
                ):

                    value = raw.iloc[
                        row_index,
                        col_index
                    ]

                    text = normalize_text(
                        value
                    )

                    if not text:

                        continue


                    # Ignore generic Power headings
                    if text in [
                        "power [kw]",
                        "power",
                        "power kw",
                        "reference power",
                        "ref power",
                        "theoretical power"
                    ]:

                        continue


                    if (
                        "power" in text
                        and "den" not in text
                        and len(text) < 30
                    ):

                        continue


                    possible_site_names.append(
                        (
                            row_index,
                            str(value).strip()
                        )
                    )


                # -------------------------------------------------
                # COMPARE AGAINST SELECTED SITE
                # -------------------------------------------------

                for (
                    row_index,
                    reference_site_name
                ) in possible_site_names:

                    if site_names_match(
                        site_name,
                        reference_site_name
                    ):

                        site_candidates.append(
                            {
                                "sheet": sheet,
                                "column_index": col_index,
                                "site_name": reference_site_name,
                                "site_row": row_index
                            }
                        )


        # =================================================
        # SITE NOT FOUND
        # =================================================

        if not site_candidates:

            # ---------------------------------------------
            # Create a useful diagnostic list
            # ---------------------------------------------

            detected_sites = []


            try:

                for sheet in excel_file.sheet_names:

                    raw = pd.read_excel(
                        REF_FILE_PATH,
                        sheet_name=sheet,
                        header=None
                    )

                    if raw.empty:
                        continue

                    raw = raw.dropna(
                        axis=0,
                        how="all"
                    )

                    raw = raw.dropna(
                        axis=1,
                        how="all"
                    )

                    for col_index in range(
                        1,
                        len(raw.columns)
                    ):

                        for row_index in range(
                            min(10, len(raw))
                        ):

                            value = raw.iloc[
                                row_index,
                                col_index
                            ]

                            text = normalize_text(
                                value
                            )

                            if not text:
                                continue

                            if (
                                "power" in text
                                and len(text) < 30
                            ):
                                continue

                            # A possible site name generally
                            # contains letters and is not numeric.
                            try:

                                float(text)

                                is_numeric = True

                            except Exception:

                                is_numeric = False


                            if not is_numeric:

                                detected_sites.append(
                                    str(value).strip()
                                )

            except Exception:

                pass


            # Remove duplicates
            detected_sites = list(
                dict.fromkeys(
                    detected_sites
                )
            )


            message = (
                "Reference curve could not be detected "
                f"for site:\n\n{site_name}\n\n"
                "The new reference Excel format expects:\n\n"
                "Column A  → Wind Speed [m/s]\n"
                "Other columns → Power [kW]\n"
                "A row below the Power heading → Site name\n\n"
                "Please check the spelling of the site name "
                "between Site Add-on and the reference Excel."
            )


            if detected_sites:

                message += (
                    "\n\nDetected site names in reference Excel:\n"
                )

                for s in detected_sites:

                    message += (
                        f"\n• {s}"
                    )


            raise ValueError(
                message
            )


        # =================================================
        # SELECT MATCH
        # =================================================

        selected_candidate = (
            site_candidates[0]
        )


        sheet = (
            selected_candidate["sheet"]
        )

        power_column_index = (
            selected_candidate["column_index"]
        )

        site_row = (
            selected_candidate["site_row"]
        )

        matched_site_name = (
            selected_candidate["site_name"]
        )


        # =================================================
        # READ SELECTED SHEET
        # =================================================

        raw = pd.read_excel(
            REF_FILE_PATH,
            sheet_name=sheet,
            header=None
        )


        # =================================================
        # FIND WIND SPEED COLUMN
        # =================================================

        wind_column_index = None


        for col_index in range(
            len(raw.columns)
        ):

            for row_index in range(
                min(10, len(raw))
            ):

                value = raw.iloc[
                    row_index,
                    col_index
                ]

                text = normalize_text(
                    value
                )

                if (
                    "wind speed" in text
                    or "windspeed" in text
                    or "wind_speed" in text
                ):

                    wind_column_index = (
                        col_index
                    )

                    break

            if wind_column_index is not None:

                break


        # -------------------------------------------------
        # FALLBACK: FIRST COLUMN
        # -------------------------------------------------

        if wind_column_index is None:

            wind_column_index = 0


        # =================================================
        # EXTRACT WIND + POWER
        # =================================================

        reference_data = pd.DataFrame(
            {
                "WindSpeed": raw.iloc[
                    site_row + 1:,
                    wind_column_index
                ],

                "Power": raw.iloc[
                    site_row + 1:,
                    power_column_index
                ]
            }
        )


        # =================================================
        # CONVERT NUMERIC
        # =================================================

        reference_data["WindSpeed"] = pd.to_numeric(
            reference_data["WindSpeed"],
            errors="coerce"
        )

        reference_data["Power"] = pd.to_numeric(
            reference_data["Power"],
            errors="coerce"
        )


        # =================================================
        # REMOVE INVALID ROWS
        # =================================================

        reference_data = (
            reference_data
            .dropna(
                subset=[
                    "WindSpeed",
                    "Power"
                ]
            )
        )


        # =================================================
        # REMOVE NEGATIVE VALUES
        # =================================================

        reference_data = reference_data[
            reference_data["WindSpeed"] >= 0
        ]

        reference_data = reference_data[
            reference_data["Power"] >= 0
        ]


        # =================================================
        # REMOVE DUPLICATE WIND SPEEDS
        # =================================================

        reference_data = (
            reference_data
            .sort_values(
                "WindSpeed"
            )
            .drop_duplicates(
                subset=[
                    "WindSpeed"
                ],
                keep="first"
            )
        )


        if len(reference_data) < 5:

            raise ValueError(
                f"Reference data for site "
                f"'{matched_site_name}' contains "
                "fewer than 5 valid Wind Speed / Power rows."
            )


        # =================================================
        # LIMIT WIND RANGE
        # =================================================

        reference_data = reference_data[
            (
                reference_data["WindSpeed"]
                <= MAX_REFERENCE_WIND
            )
        ]


        # =================================================
        # CREATE STANDARD 0.5 m/s BINS
        # =================================================

        min_available_wind = (
            float(
                reference_data[
                    "WindSpeed"
                ].min()
            )
        )

        max_available_wind = (
            float(
                reference_data[
                    "WindSpeed"
                ].max()
            )
        )


        min_wind = max(
            REFERENCE_START_WIND,
            min_available_wind
        )

        max_wind = min(
            REFERENCE_END_WIND,
            max_available_wind
        )


        if max_wind <= min_wind:

            raise ValueError(
                "Reference wind-speed range is invalid."
            )


        wind_bins = np.arange(
            min_wind,
            max_wind + BIN_SIZE / 2,
            BIN_SIZE
        )


        # =================================================
        # INTERPOLATE POWER
        # =================================================

        ref_power = np.interp(
            wind_bins,
            reference_data[
                "WindSpeed"
            ].values,
            reference_data[
                "Power"
            ].values
        )


        # =================================================
        # CREATE REFERENCE CURVE
        # =================================================

        ref_curve = pd.DataFrame(
            {
                "WindBin": np.round(
                    wind_bins,
                    6
                ),

                "RefPower": ref_power
            }
        )


        # =================================================
        # FINAL VALIDATION
        # =================================================

        if ref_curve.empty:

            raise ValueError(
                "Reference curve is empty."
            )


        # =================================================
        # RETURN
        # =================================================

        return ref_curve


    # =====================================================
    # LOAD REFERENCE
    # =====================================================

    try:

        ref_curve = load_reference(
            site
        )

    except Exception as e:

        st.error(
            "Reference curve could not be loaded."
        )

        st.code(
            str(e)
        )

        st.stop()


    # =====================================================
    # SHOW REFERENCE INFORMATION
    # =====================================================

    with st.expander(
        "Reference Curve Information"
    ):

        st.write(
            f"Selected Site: **{site}**"
        )

        st.write(
            f"Reference points: "
            f"**{len(ref_curve)}**"
        )

        st.write(
            f"Wind range: "
            f"**{ref_curve['WindBin'].min():.1f} "
            f"to "
            f"{ref_curve['WindBin'].max():.1f} m/s**"
        )

        st.dataframe(
            ref_curve,
            use_container_width=True
        )


    # =====================================================
    # PROCESS TURBINE
    # =====================================================

    def process_turbine(t):

        df_t = df[
            df["Name"] == t
        ].copy()


        # -------------------------------------------------
        # SCADA FILTER
        # -------------------------------------------------

        df_t = df_t[
            (df_t[wind_col] >= 3)
            &
            (df_t[wind_col] <= 25)
            &
            (df_t[power_col] > 0)
        ]


        # -------------------------------------------------
        # PITCH FILTER
        # -------------------------------------------------

        if pitch_col != "_Pitch_Default":

            df_t = df_t[
                (df_t[pitch_col] >= -5)
                &
                (df_t[pitch_col] <= 5)
            ]


        if len(df_t) < 30:

            return None


        # -------------------------------------------------
        # STANDARD DEVIATION
        # -------------------------------------------------

        std_dev = (
            df_t[power_col].std()
        )


        # -------------------------------------------------
        # WIND BIN
        # -------------------------------------------------

        df_t["WindBin"] = (
            np.floor(
                df_t[wind_col]
                / BIN_SIZE
            )
            * BIN_SIZE
        ).round(6)


        # -------------------------------------------------
        # ACTUAL CURVE
        # -------------------------------------------------

        actual = (
            df_t
            .groupby(
                "WindBin"
            )
            .agg(
                AvgPower=(
                    power_col,
                    "mean"
                )
            )
            .reset_index()
        )


        # =================================================
        # MERGE ACTUAL + REFERENCE
        # =================================================

        merged = ref_curve.merge(
            actual,
            on="WindBin",
            how="left"
        )


        # =================================================
        # SMOOTH ACTUAL
        # =================================================

        valid = (
            merged["AvgPower"]
            .notna()
        )


        if valid.sum() >= 7:

            values = (
                merged.loc[
                    valid,
                    "AvgPower"
                ].values
            )


            window = min(
                7,
                len(values)
            )


            if window % 2 == 0:

                window -= 1


            if window >= 5:

                try:

                    smoothed = (
                        savgol_filter(
                            values,
                            window,
                            2
                        )
                    )

                    merged.loc[
                        valid,
                        "AvgPower"
                    ] = smoothed

                except Exception:

                    pass


        # =================================================
        # DEVIATION
        # =================================================

        merged["Deviation_%"] = np.where(

            merged["RefPower"] != 0,

            (
                (
                    merged["AvgPower"]
                    - merged["RefPower"]
                )
                /
                merged["RefPower"]
            )
            * 100,

            np.nan
        )


        # =================================================
        # AVERAGE DEVIATION
        # =================================================

        avg_dev = (
            merged["Deviation_%"]
            .mean(
                skipna=True
            )
        )


        return (
            df_t,
            merged,
            avg_dev,
            std_dev
        )


    # =====================================================
    # PLOT GRAPH
    # =====================================================

    def plot_graph(
        df_t,
        merged,
        title,
        dev
    ):

        if (
            dev is None
            or pd.isna(dev)
        ):

            title_color = "gray"

        elif -2 <= dev <= 2:

            title_color = "green"

        elif dev < -2:

            title_color = "orange"

        else:

            title_color = "red"


        fig = go.Figure()


        # =================================================
        # SCATTER
        # =================================================

        fig.add_trace(
            go.Scatter(
                x=df_t[wind_col],
                y=df_t[power_col],
                mode="markers",

                marker=dict(
                    size=4,
                    opacity=0.35,
                    color="rgba(30, 144, 255, 0.55)"
                ),

                name="SCADA Data"
            )
        )


        # =================================================
        # REFERENCE CURVE
        # =================================================

        fig.add_trace(
            go.Scatter(
                x=merged["WindBin"],
                y=merged["RefPower"],
                mode="lines",

                line=dict(
                    dash="dash",
                    width=3,
                    color="red"
                ),

                name="Reference"
            )
        )


        # =================================================
        # ACTUAL CURVE
        # =================================================

        fig.add_trace(
            go.Scatter(
                x=merged["WindBin"],
                y=merged["AvgPower"],
                mode="lines+markers",

                line=dict(
                    width=4,
                    color="green"
                ),

                marker=dict(
                    size=6,
                    color="green"
                ),

                name="Actual"
            )
        )


        # =================================================
        # LAYOUT
        # =================================================

        fig.update_layout(

            title=dict(

                text=(
                    f"{title} "
                    f"(Dev: "
                    f"{round(dev, 2)}%)"
                ),

                font=dict(
                    color=title_color
                )
            ),

            xaxis=dict(
                title="Wind Speed (m/s)",
                range=[
                    2.5,
                    25
                ],
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=True
            ),

            yaxis=dict(
                title="Power (kW)",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=True
            ),

            height=500,

            hovermode="x unified",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )


        return fig


    # =====================================================
    # COMMENTS
    # =====================================================

    def generate_comment(dev):

        if (
            dev is None
            or pd.isna(dev)
        ):

            return (
                "Data not available"
            )


        dev = round(
            float(dev),
            2
        )


        if dev < -72:

            return (
                f"Dev: {dev}% → "
                "Extreme issue "
                "(The Data unreliable)"
            )


        elif dev < -10:

            return (
                f"Dev: {dev}% → "
                "Severe underperformance "
                "(Blade/Dust/Yaw issue)"
            )


        elif dev < -2:

            return (
                f"Dev: {dev}% → "
                "Underperformance "
                "(Control/availability)"
            )


        elif dev > 72:

            return (
                f"Dev: {dev}% → "
                "Abnormal high "
                "(Sensor/Data issue)"
            )


        elif dev > 8:

            return (
                f"Dev: {dev}% → "
                "High overperformance"
            )


        elif dev > 2:

            return (
                f"Dev: {dev}% → "
                "Slight overperformance"
            )


        else:

            return (
                f"Dev: {dev}% → "
                "Normal performance"
            )


    # =====================================================
    # STATUS
    # =====================================================

    def get_status(dev):

        if dev is None or pd.isna(dev):

            return "Issue"

        if -2 <= dev <= 2:

            return "Normal"

        elif 2 < dev <= 8:

            return "Slight Over"

        elif dev > 8:

            return "High Over"

        elif -10 <= dev < -2:

            return "Under"

        elif dev < -10:

            return "High Under"

        return "Issue"


    # =====================================================
    # TURBINE SELECTION
    # =====================================================

    turbines = (
        df["Name"]
        .dropna()
        .unique()
    )


    if mode == "Single Turbine":

        selected = st.sidebar.selectbox(
            "Select Turbine",
            turbines,
            key="single_turbine"
        )

        turbines_to_show = [
            selected
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


    # =====================================================
    # DISPLAY
    # =====================================================

    cols = st.columns(2)

    results = []

    figures = []

    i = 0


    for t in turbines_to_show:

        res = process_turbine(
            t
        )

        if res is None:

            continue


        (
            df_t,
            merged,
            dev,
            std
        ) = res


        # -------------------------------------------------
        # GRAPH
        # -------------------------------------------------

        fig = plot_graph(
            df_t,
            merged,
            t,
            dev
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
                generate_comment(
                    dev
                )
            )


            st.caption(
                f"Power Standard Deviation: "
                f"{round(std, 2)} kW"
            )


        comment = (
            generate_comment(
                dev
            )
        )


        figures.append(
            (
                t,
                fig,
                comment
            )
        )


        # =================================================
        # STATUS
        # =================================================

        status = get_status(
            dev
        )


        results.append(
            {
                "Turbine": t,

                "Deviation_%": round(
                    float(dev),
                    2
                )
                if pd.notna(dev)
                else np.nan,

                "Status": status,

                "Comment": comment
            }
        )


        i += 1


    # =====================================================
    # RANKING TABLE
    # =====================================================

    st.subheader(
        "Turbine Ranking"
    )


    results_df = pd.DataFrame(
        results
    )


    if not results_df.empty:

        results_df = (
            results_df
            .sort_values(
                by="Deviation_%",
                na_position="last"
            )
        )


        # -------------------------------------------------
        # ROW COLOR
        # -------------------------------------------------

        def color_row(row):

            if row["Status"] == "Normal":

                return [
                    "background-color: #ccffcc"
                ] * len(row)


            elif row["Status"] == "Slight Over":

                return [
                    "background-color: #66ff66"
                ] * len(row)


            elif row["Status"] == "High Over":

                return [
                    "background-color: #009933"
                ] * len(row)


            elif row["Status"] == "Under":

                return [
                    "background-color: #ffcc66"
                ] * len(row)


            elif row["Status"] == "High Under":

                return [
                    "background-color: #ff6666"
                ] * len(row)


            else:

                return [
                    "background-color: #cccccc"
                ] * len(row)


        styled_table = (
            results_df.style
            .apply(
                color_row,
                axis=1
            )
        )


        st.dataframe(
            styled_table,
            use_container_width=True
        )


    else:

        st.info(
            "No turbine has enough valid data "
            "for the selected date range."
        )


    # =====================================================
    # PDF REPORT
    # =====================================================

    try:

        pdf_buffer = io.BytesIO()

        pdf = canvas.Canvas(
            pdf_buffer,
            pagesize=landscape(A4)
        )

        width, height = landscape(A4)


        # =================================================
        # LOGO
        # =================================================

        if os.path.exists(
            logo_path
        ):

            try:

                pdf.drawImage(
                    logo_path,
                    30,
                    height - 80,
                    width=120,
                    height=40,
                    preserveAspectRatio=True
                )

            except Exception:

                pass


        # =================================================
        # TITLE
        # =================================================

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


        y = height - 120


        # =================================================
        # GRAPH REPORT
        # =================================================

        for (
            turbine,
            fig,
            comment
        ) in figures:

            if KALEIDO_AVAILABLE:

                try:

                    img = fig.to_image(
                        format="png"
                    )

                    img_reader = ImageReader(
                        io.BytesIO(img)
                    )


                    if y < 260:

                        pdf.showPage()

                        y = height - 60


                    pdf.drawImage(
                        img_reader,
                        30,
                        y - 220,
                        width=360,
                        height=200
                    )


                    pdf.setFont(
                        "Helvetica-Bold",
                        11
                    )

                    pdf.drawString(
                        420,
                        y - 40,
                        str(turbine)
                    )


                    pdf.setFont(
                        "Helvetica",
                        10
                    )

                    pdf.drawString(
                        420,
                        y - 60,
                        comment
                    )


                    y -= 240


                except Exception:

                    pass


        # =================================================
        # RANKING PAGE
        # =================================================

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
            10
        )


        if not results_df.empty:

            for _, row in results_df.iterrows():

                line = (
                    f"{row['Turbine']} | "
                    f"{row['Deviation_%']} % | "
                    f"{row['Status']}"
                )


                pdf.drawString(
                    40,
                    y,
                    line
                )


                y -= 20


                if y < 40:

                    pdf.showPage()

                    y = height - 40

                    pdf.setFont(
                        "Helvetica",
                        10
                    )


        pdf.save()

        pdf_buffer.seek(0)


        # =================================================
        # DOWNLOAD
        # =================================================

        st.download_button(

            label=(
                "Download Full Dashboard "
                "Report (PDF)"
            ),

            data=pdf_buffer.getvalue(),

            file_name=(
                "WindFarm_Full_Report.pdf"
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
