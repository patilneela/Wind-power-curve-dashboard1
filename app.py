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


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    layout="wide",
    page_title="Power Curve Analytics Report"
)


# =========================
# SIMPLE LOCK (single user)
# =========================
def login_gate():

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("Login Required")

    with st.form(
        "login_form",
        clear_on_submit=False
    ):

        username = st.text_input(
            "Username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        submitted = st.form_submit_button(
            "Login"
        )

    if submitted:

        cfg = st.secrets.get(
            "auth",
            {}
        )

        ok = (
            username == cfg.get("username")
            and
            password == cfg.get("password")
        )

        if ok:

            st.session_state.authenticated = True
            st.rerun()

        else:

            st.error(
                "Invalid username or password"
            )

    st.stop()


login_gate()


# =========================
# LOGOUT
# =========================
if st.sidebar.button(
    "Logout",
    key="logout_btn"
):

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
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# -------------------------------------------------
# NEW REFERENCE FILE
# -------------------------------------------------
REF_FILE_PATH = os.path.join(
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

BIN_SIZE = 0.5


# ==========================================================
# HELPERS
# ==========================================================

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


def compute_preset_range(
    preset: str,
    today_ts: pd.Timestamp
):

    """
    Returns:
        (start_day, end_day)

    Both dates are inclusive.
    """

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
            -
            timedelta(
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
            -
            timedelta(days=1)
        )

        start = (
            last_month_end
            .replace(day=1)
        )

        return (
            start,
            last_month_end
        )

    return None


# ==========================================================
# REFERENCE SITE NAME NORMALIZATION
# ==========================================================

def normalize_site_name(value):

    """
    Normalizes site names so that small differences in
    spaces, hyphens and dash characters do not prevent
    matching.

    Example:

    Renew-4   -Kudligi- KA
    Renew-4 -Kudligi- KA

    will be treated as the same site.
    """

    if pd.isna(value):

        return ""

    text = str(value)

    text = text.replace(
        "\xa0",
        " "
    )

    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "—",
        "-"
    )

    # Convert multiple spaces to one
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip().lower()

    # Remove spaces around hyphens
    text = re.sub(
        r"\s*-\s*",
        "-",
        text
    )

    return text


def compact_site_name(value):

    """
    Removes spaces, hyphens and punctuation.

    Used as a second-level site matching method.
    """

    text = normalize_site_name(
        value
    )

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text
    )



# ==========================================================
# TURBINE NAME SORTING
# ==========================================================

def natural_sort_key(value):
    """Sort turbine names naturally: ANB-2 before ANB-12."""
    text = str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


# ==========================================================
# LOGO
# ==========================================================

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


# ==========================================================
# TITLE
# ==========================================================

st.title(
    "Power Curve Analytics Report"
)


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
        "India_Hero_Doni",

        # -----------------------------------------------
        # NEW REFERENCE FILE SITES
        # -----------------------------------------------
        "CleanMax -Gujarat (*Den-1.142)",
        "Renew-4   -Kudligi- KA (*Den-1.076)",
        "Renew-4-Otha   - GJ (*Den-1.153)",
        "Renew   -Pithalur-GJ  (*Den-1.153)",
        "CleanMax-Jagalur-KA   (*Den-1.070)",
        "Fourthpartner-Ottapidaram-TN  (*Den-1.145)",
        "Sembcorp   Tuticorin-TN  (*Den-1.145)",
        "AMGPEL Kurnool",
        "JSW Sandur KA",
        "Fourthpartner   Kudligi KA",
        "Ayana Amreli GJ",
        "Sprng Mulanur TN   (1.125Kg/m3)",
        "ACME Shapur GJ   (1.136Kg/m3)",
        "Cleanmax Honavad   KA (1.105Kg/m3)",
        "Renfra trichy TN",
        "NSL_AP (1.089 Kg/m3)"
    ]
}


# ==========================================================
# LOAD SITE MASTER
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

            c_lower = c.lower()

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
            or
            cap_col is None
        ):

            st.warning(
                "site_master file found but columns "
                "not recognized. Required: Site + Capacity_MW."
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
            "Failed to read site_master. "
            "Using default site list."
        )

        st.code(
            str(e)
        )

        return capacity


SITE_CAPACITY = load_site_capacity()


# ==========================================================
# AUTOMATICALLY ADD SITES FOUND IN NEW REFERENCE FILE
# ==========================================================

@st.cache_data
def clean_site_display_name(value):
    """Clean site names from Excel while preserving useful details."""
    if pd.isna(value):
        return ""
    text=str(value).replace("\xa0"," ").replace("\r"," ").replace("\n"," ")
    return re.sub(r"\s+"," ",text).strip()


def _is_power_header(value):
    text=normalize_site_name(value)
    return "power" in text and ("kw" in text or text=="power")


def _is_wind_header(value):
    text=normalize_site_name(value)
    return "wind speed" in text or "windspeed" in text or "wind_speed" in text


def _excel_source(reference_path):
    if isinstance(reference_path,(bytes,bytearray)):
        return io.BytesIO(reference_path)
    return reference_path


@st.cache_data
def get_reference_site_names(reference_path):
    """Detect sites from Power Curve: Power [kW] header, site directly below."""
    if reference_path is None:
        return []
    try:
        source=_excel_source(reference_path)
        xls=pd.ExcelFile(source)
        sheets=["Power Curve"] if "Power Curve" in xls.sheet_names else xls.sheet_names
        sites=[]
        for sheet in sheets:
            raw=pd.read_excel(_excel_source(reference_path),sheet_name=sheet,header=None)
            if raw.empty: continue
            wind_row=None
            for r in range(min(15,len(raw))):
                if any(_is_wind_header(v) for v in raw.iloc[r].tolist()):
                    wind_row=r; break
            if wind_row is None: continue
            power_row=None
            for r in range(max(0,wind_row-2),min(len(raw),wind_row+4)):
                if any(_is_power_header(v) for v in raw.iloc[r].tolist()):
                    power_row=r; break
            if power_row is None: continue
            site_row=power_row+1
            if site_row>=len(raw): continue
            for c in range(raw.shape[1]):
                if not _is_power_header(raw.iloc[power_row,c]): continue
                site=clean_site_display_name(raw.iloc[site_row,c])
                if not site or _is_power_header(site) or _is_wind_header(site): continue
                if pd.notna(pd.to_numeric(site,errors="coerce")): continue
                sites.append(site)
            if sites and sheet=="Power Curve": break
        unique=[]; seen=set()
        for site in sites:
            site=clean_site_display_name(site); key=normalize_site_name(site)
            if key and key not in seen:
                seen.add(key); unique.append(site)
        return unique
    except Exception:
        return []


# Add reference sites automatically
REFERENCE_SITES = get_reference_site_names(
    REF_FILE_PATH
)

existing_capacity_keys = {
    normalize_site_name(site)
    for site in SITE_CAPACITY.keys()
}

for ref_site in REFERENCE_SITES:

    normalized_ref = normalize_site_name(
        ref_site
    )

    if (
        normalized_ref
        not in existing_capacity_keys
    ):

        SITE_CAPACITY[
            ref_site
        ] = 3.3

        existing_capacity_keys.add(
            normalized_ref
        )


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
# TAB: ADMIN
# ==========================================================

with tab_admin:

    st.subheader(
        "Site Add-on"
    )

    st.divider()

    # ======================================================
    # REFERENCE EXCEL
    # ======================================================

    st.markdown(
        "## 1) Reference Excel (reference.xlsx)"
    )

    if os.path.exists(
        REF_FILE_PATH
    ):

        st.success(
            f"Reference file found: "
            f"`{os.path.basename(REF_FILE_PATH)}`"
        )

    else:

        st.warning(
            "Reference file missing. "
            "Upload a reference Excel to enable "
            "power curve reference comparison."
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
                    "Reference Excel saved/replaced."
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

    # ======================================================
    # REFERENCE PREVIEW
    # ======================================================

    if os.path.exists(
        REF_FILE_PATH
    ):

        with st.expander(
            "Preview reference.xlsx (first 30 rows)"
        ):

            try:

                tmp = pd.read_excel(
                    REF_FILE_PATH,
                    header=None
                )

                st.dataframe(
                    tmp.head(30),
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    "Unable to read reference.xlsx"
                )

                st.code(
                    str(e)
                )

    st.divider()

    # ======================================================
    # SITE MASTER
    # ======================================================

    st.markdown(
        "## 2) Site Master (site_master.xlsx / site_master.csv)"
    )

    existing_sm = get_site_master_path()

    if existing_sm:

        st.success(
            f"Site Master found: "
            f"`{os.path.basename(existing_sm)}`"
        )

    else:

        st.info(
            "No Site Master file found. "
            "Dashboard will use the hardcoded default site list."
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
                    "Please choose a .xlsx or .csv file first."
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
                        and
                        os.path.exists(
                            SITE_MASTER_XLSX
                        )
                    ):

                        os.remove(
                            SITE_MASTER_XLSX
                        )

                    if (
                        target != SITE_MASTER_CSV
                        and
                        os.path.exists(
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
                        f"Site Master saved as "
                        f"`{os.path.basename(target)}`"
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

    existing_sm = get_site_master_path()

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
                    "Unable to read Site Master file."
                )

                st.code(
                    str(e)
                )


# ==========================================================
# TAB: DASHBOARD
# ==========================================================

with tab_dashboard:

    # ======================================================
    # UPLOAD SCADA + REFERENCE FILE
    # ======================================================

    st.sidebar.subheader(
        "Upload Input Files"
    )

    uploaded_file = st.sidebar.file_uploader(
        "1. Upload SCADA CSV",
        type=["csv"],
        key="scada_upload"
    )

    uploaded_reference = st.sidebar.file_uploader(
        "2. Upload Reference Excel",
        type=["xlsx"],
        key="dashboard_reference_upload"
    )

    if uploaded_file is None:
        st.warning("Please upload the SCADA CSV file.")
        st.stop()

    if uploaded_reference is None:
        st.warning("Please upload the updated Reference Excel file.")
        st.stop()

    reference_bytes = uploaded_reference.getvalue()

    # Detect sites directly from the uploaded reference workbook.
    uploaded_reference_sites = get_reference_site_names(reference_bytes)

    if not uploaded_reference_sites:
        st.error("No site names could be detected in the uploaded reference Excel.")
        st.info("Expected format: the Power Curve sheet must contain 'Power [kW]' headers with the site name directly below each Power [kW] column.")
        st.stop()

    # Add newly uploaded reference sites to capacity map.
    for ref_site in uploaded_reference_sites:
        if normalize_site_name(ref_site) not in {
            normalize_site_name(x) for x in SITE_CAPACITY.keys()
        }:
            SITE_CAPACITY[ref_site] = 3.3

    # ======================================================
    # SITE + MODE
    # ======================================================

    site = st.sidebar.selectbox(
        "Select Site",
        uploaded_reference_sites,
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

        df_local.columns = (
            df_local.columns
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------
        # NAME
        # --------------------------------------------------

        if "Name" not in df_local.columns:

            raise ValueError(
                "SCADA CSV must contain a "
                "'Name' column for turbine identifier."
            )

        # --------------------------------------------------
        # WIND COLUMN
        # --------------------------------------------------

        wind_candidates = [
            c
            for c in df_local.columns
            if "wind" in c.lower()
        ]

        if not wind_candidates:

            raise ValueError(
                "Wind speed column not found in SCADA CSV."
            )

        wind_col = wind_candidates[0]

        # --------------------------------------------------
        # POWER COLUMN
        # --------------------------------------------------

        power_candidates = [
            c
            for c in df_local.columns
            if (
                "power" in c.lower()
                or
                "active" in c.lower()
            )
        ]

        if not power_candidates:

            raise ValueError(
                "Power column not found in SCADA CSV."
            )

        power_col = power_candidates[0]

        # --------------------------------------------------
        # TIME COLUMN
        # --------------------------------------------------

        time_candidates = [
            c
            for c in df_local.columns
            if (
                "time" in c.lower()
                or
                "date" in c.lower()
            )
        ]

        if not time_candidates:

            raise ValueError(
                "Time/date column not found in SCADA CSV."
            )

        time_col = time_candidates[0]

        # --------------------------------------------------
        # PITCH COLUMN
        # --------------------------------------------------

        pitch_candidates = [
            c
            for c in df_local.columns
            if "pitch" in c.lower()
        ]

        if pitch_candidates:

            pitch_col = pitch_candidates[0]

            df_local[
                pitch_col
            ] = pd.to_numeric(
                df_local[pitch_col],
                errors="coerce"
            )

        else:

            # If pitch is not available,
            # use a neutral default value.
            pitch_col = "_Pitch_Default"

            df_local[
                pitch_col
            ] = 0.0

        # --------------------------------------------------
        # DATA TYPE CONVERSION
        # --------------------------------------------------

        df_local[
            time_col
        ] = pd.to_datetime(
            df_local[time_col],
            errors="coerce"
        )

        df_local[
            wind_col
        ] = pd.to_numeric(
            df_local[wind_col],
            errors="coerce"
        )

        df_local[
            power_col
        ] = pd.to_numeric(
            df_local[power_col],
            errors="coerce"
        )

        # --------------------------------------------------
        # CLEAN TURBINE NAMES
        # --------------------------------------------------

        df_local["Name"] = (
            df_local["Name"]
            .astype(str)
            .str.replace("\\u00a0", " ", regex=False)
            .str.strip()
        )

        # IMPORTANT:
        # Keep the complete turbine list BEFORE removing rows
        # with invalid Wind/Power/Time values. This prevents
        # turbines from disappearing from the dashboard merely
        # because their data is incomplete.
        all_turbines_local = [
            x for x in df_local["Name"].dropna().unique()
            if str(x).strip() and str(x).strip().lower() != "nan"
        ]
        all_turbines_local = sorted(
            set(all_turbines_local),
            key=natural_sort_key
        )

        # --------------------------------------------------
        # DROP INVALID MEASUREMENT ROWS
        # --------------------------------------------------

        df_local = df_local.dropna(
            subset=[
                wind_col,
                power_col,
                time_col
            ]
        )

        return (
            df_local,
            wind_col,
            power_col,
            time_col,
            pitch_col,
            all_turbines_local
        )

    # ======================================================
    # LOAD SCADA
    # ======================================================

    with st.spinner(
        "Loading SCADA file..."
    ):

        try:

            (
                df,
                wind_col,
                power_col,
                time_col,
                pitch_col,
                all_turbines
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
            "SCADA file has no valid rows after parsing."
        )

        st.stop()

    # ======================================================
    # DATE FILTER
    # ======================================================

    st.sidebar.markdown(
        "### Date Range"
    )

    max_ts = df[
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

        st.session_state[
            "manual_start_date"
        ] = DEFAULT_START

    if (
        "manual_end_date"
        not in st.session_state
    ):

        st.session_state[
            "manual_end_date"
        ] = DEFAULT_END

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
            value=st.session_state[
                "manual_start_date"
            ],
            key="manual_start_date"
        )

        end_day = st.sidebar.date_input(
            "End Date",
            value=st.session_state[
                "manual_end_date"
            ],
            key="manual_end_date"
        )

    elif date_option == "Clear":

        st.session_state[
            "manual_start_date"
        ] = DEFAULT_START

        st.session_state[
            "manual_end_date"
        ] = DEFAULT_END

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
            "for the selected date range."
        )

        st.stop()

    # ======================================================
    # HEADER
    # ======================================================

    # IMPORTANT:
    # Use the complete turbine list detected from the SCADA
    # file, not only turbines that survived the selected date
    # range or the power-curve filters.
    num_turbines = len(all_turbines)

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

    # ======================================================
    # LOAD NEW REFERENCE FILE
    # ======================================================

    @st.cache_data
    def load_reference(reference_path, site_name):
        """Load selected site's curve from the uploaded Power Curve sheet."""
        try:
            xls=pd.ExcelFile(_excel_source(reference_path))
        except Exception as e:
            raise ValueError(f"Unable to open reference Excel: {e}")

        target=normalize_site_name(site_name)
        compact=compact_site_name(site_name)
        detected=[]
        sheets=["Power Curve"] if "Power Curve" in xls.sheet_names else xls.sheet_names

        for sheet in sheets:
            try:
                raw=pd.read_excel(_excel_source(reference_path),sheet_name=sheet,header=None)
            except Exception:
                continue
            if raw.empty: continue

            wind_col=None; wind_row=None
            for r in range(min(15,len(raw))):
                for c in range(raw.shape[1]):
                    if _is_wind_header(raw.iloc[r,c]):
                        wind_col=c; wind_row=r; break
                if wind_col is not None: break
            if wind_col is None: continue

            power_row=None
            for r in range(max(0,wind_row-2),min(len(raw),wind_row+4)):
                if any(_is_power_header(v) for v in raw.iloc[r].tolist()):
                    power_row=r; break
            if power_row is None: continue

            site_row=power_row+1
            pairs=[]
            if site_row<len(raw):
                for c in range(raw.shape[1]):
                    if not _is_power_header(raw.iloc[power_row,c]): continue
                    site=clean_site_display_name(raw.iloc[site_row,c])
                    if not site or _is_power_header(site) or _is_wind_header(site): continue
                    if pd.notna(pd.to_numeric(site,errors="coerce")): continue
                    pairs.append((site,c))
            detected.extend([s for s,_ in pairs])

            site_col=None
            for site,c in pairs:
                if normalize_site_name(site)==target or compact_site_name(site)==compact:
                    site_col=c; break
            if site_col is None: continue

            data_start=max(wind_row,power_row,site_row)+1
            ref=raw.iloc[data_start:,[wind_col,site_col]].copy()
            ref.columns=["WindSpeed","RefPower"]
            ref["WindSpeed"]=pd.to_numeric(ref["WindSpeed"],errors="coerce")
            ref["RefPower"]=pd.to_numeric(ref["RefPower"],errors="coerce")
            ref=ref.dropna(subset=["WindSpeed","RefPower"])
            ref=ref[(ref["WindSpeed"]>=0)&(ref["RefPower"]>=0)]
            if ref.empty: continue
            ref=ref.sort_values("WindSpeed").drop_duplicates("WindSpeed",keep="first")
            ref=ref[(ref["WindSpeed"]>=3)&(ref["WindSpeed"]<=25)]
            if len(ref)<5: continue

            min_wind=max(3.0,float(ref["WindSpeed"].min()))
            max_wind=min(25.0,float(ref["WindSpeed"].max()))
            bins=np.round(np.arange(min_wind,max_wind+BIN_SIZE/2,BIN_SIZE),6)
            power=np.interp(bins,ref["WindSpeed"].values,ref["RefPower"].values)
            return pd.DataFrame({"WindBin":bins,"RefPower":power})

        detected=list(dict.fromkeys(clean_site_display_name(x) for x in detected if clean_site_display_name(x)))
        raise ValueError(
            "Selected site was not found in the uploaded reference Excel.\n\n"
            f"Selected site:\n{site_name}\n\nDetected sites:\n"+
            ("\n".join(f"- {x}" for x in detected) if detected else "No site names were detected in the Power Curve sheet.")
        )

    # ======================================================
    # LOAD REFERENCE
    # ======================================================

    try:

        ref_curve = load_reference(
            reference_bytes,
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

    # ======================================================
    # PROCESS TURBINE
    # ======================================================

    def process_turbine(t):

        # Always start from the currently selected date range.
        df_t_raw = df[
            df["Name"] == t
        ].copy()

        raw_rows = len(df_t_raw)

        if raw_rows == 0:
            return {
                "turbine": t,
                "df_t": df_t_raw,
                "raw_rows": 0,
                "valid_rows": 0,
                "status": "No Data",
                "reason": (
                    "No SCADA rows are available for this turbine "
                    "in the selected date range."
                ),
                "merged": None,
                "dev": np.nan,
                "std": np.nan
            }

        # --------------------------------------------------
        # SAME POWER-CURVE FILTERING LOGIC
        # --------------------------------------------------

        df_t = df_t_raw[
            (df_t_raw[wind_col] >= 3)
            &
            (df_t_raw[wind_col] <= 25)
            &
            (df_t_raw[power_col] > 0)
            &
            (df_t_raw[pitch_col] >= -5)
            &
            (df_t_raw[pitch_col] <= 5)
        ].copy()

        valid_rows = len(df_t)

        # Do NOT remove this turbine from the dashboard.
        # Return a clear data-quality status instead.
        if valid_rows < 30:
            return {
                "turbine": t,
                "df_t": df_t,
                "raw_rows": raw_rows,
                "valid_rows": valid_rows,
                "status": "Insufficient Data",
                "reason": (
                    f"Only {valid_rows} valid SCADA rows remain after "
                    "the existing wind/power/pitch filters. "
                    "At least 30 valid rows are required to calculate "
                    "the power curve."
                ),
                "merged": None,
                "dev": np.nan,
                "std": np.nan
            }

        # --------------------------------------------------
        # STANDARD DEVIATION
        # --------------------------------------------------

        std_dev = df_t[power_col].std()

        # --------------------------------------------------
        # WIND BIN
        # --------------------------------------------------

        df_t["WindBin"] = (
            np.floor(
                df_t[wind_col] / BIN_SIZE
            ) * BIN_SIZE
        ).round(6)

        # --------------------------------------------------
        # ACTUAL CURVE
        # --------------------------------------------------

        actual = (
            df_t
            .groupby("WindBin")
            .agg(
                AvgPower=(power_col, "mean")
            )
            .reset_index()
        )

        # --------------------------------------------------
        # MERGE WITH REFERENCE
        # --------------------------------------------------

        merged = ref_curve.merge(
            actual,
            on="WindBin",
            how="left"
        )

        # --------------------------------------------------
        # SMOOTH ACTUAL CURVE
        # --------------------------------------------------

        valid = merged["AvgPower"].notna()

        if valid.sum() >= 7:

            values = merged.loc[
                valid, "AvgPower"
            ].values

            window = min(7, len(values))

            if window % 2 == 0:
                window -= 1

            if window >= 5:

                merged.loc[
                    valid, "AvgPower"
                ] = savgol_filter(
                    values,
                    window,
                    2
                )

        # --------------------------------------------------
        # DEVIATION
        # --------------------------------------------------

        merged["Deviation_%"] = np.where(
            merged["RefPower"] != 0,
            (
                (
                    merged["AvgPower"]
                    -
                    merged["RefPower"]
                )
                /
                merged["RefPower"]
            ) * 100,
            np.nan
        )

        # --------------------------------------------------
        # AVERAGE DEVIATION
        # --------------------------------------------------

        avg_dev = merged["Deviation_%"].mean(
            skipna=True
        )

        if pd.isna(avg_dev):
            return {
                "turbine": t,
                "df_t": df_t,
                "raw_rows": raw_rows,
                "valid_rows": valid_rows,
                "status": "No Curve",
                "reason": (
                    "Valid SCADA rows are present, but there are not "
                    "enough overlapping wind-speed bins with the "
                    "reference curve to calculate deviation."
                ),
                "merged": merged,
                "dev": np.nan,
                "std": std_dev
            }

        return {
            "turbine": t,
            "df_t": df_t,
            "raw_rows": raw_rows,
            "valid_rows": valid_rows,
            "status": "Calculated",
            "reason": "",
            "merged": merged,
            "dev": float(avg_dev),
            "std": std_dev
        }


    # ======================================================
    # PLOT GRAPH
    # ======================================================

    def plot_graph(
        df_t,
        merged,
        title,
        dev,
        show_deviation=True
    ):
        """
        Larger, cleaner Plotly graph designed for the dashboard.
        Two graphs are displayed per row so wind-speed/power points
        remain readable while still keeping 6 graphs per page.
        """

        if dev is None or pd.isna(dev):
            title_color = "gray"
        elif -2 <= dev <= 2:
            title_color = "green"
        elif dev < -2:
            title_color = "orange"
        else:
            title_color = "red"

        fig = go.Figure()

        # --------------------------------------------------
        # RAW SCATTER
        # --------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=df_t[wind_col],
                y=df_t[power_col],
                mode="markers",
                marker=dict(
                    size=4.5,
                    opacity=0.30
                ),
                name="SCADA Points",
                hovertemplate=(
                    "<b>%{x:.2f} m/s</b><br>"
                    "Power: %{y:.1f} kW"
                    "<extra>SCADA</extra>"
                )
            )
        )

        # --------------------------------------------------
        # REFERENCE
        # --------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=merged["WindBin"],
                y=merged["RefPower"],
                mode="lines",
                line=dict(
                    dash="dash",
                    width=3
                ),
                name="Reference",
                hovertemplate=(
                    "<b>%{x:.2f} m/s</b><br>"
                    "Reference: %{y:.1f} kW"
                    "<extra>Reference</extra>"
                )
            )
        )

        # --------------------------------------------------
        # ACTUAL
        # --------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=merged["WindBin"],
                y=merged["AvgPower"],
                mode="lines+markers",
                line=dict(
                    width=3
                ),
                marker=dict(
                    size=6
                ),
                name="Actual",
                connectgaps=False,
                hovertemplate=(
                    "<b>%{x:.2f} m/s</b><br>"
                    "Actual: %{y:.1f} kW"
                    "<extra>Actual</extra>"
                )
            )
        )

        # --------------------------------------------------
        # READABLE AXIS RANGE
        # --------------------------------------------------

        x_values = []

        if len(df_t):
            x_values.extend(
                pd.to_numeric(
                    df_t[wind_col],
                    errors="coerce"
                ).dropna().tolist()
            )

        if merged is not None and not merged.empty:
            x_values.extend(
                pd.to_numeric(
                    merged["WindBin"],
                    errors="coerce"
                ).dropna().tolist()
            )

        if x_values:
            x_max = float(max(x_values))
            x_min = float(min(x_values))
            x_left = max(2.5, x_min - 0.4)
            x_right = min(25.5, max(12.0, x_max + 0.8))
        else:
            x_left, x_right = 2.5, 12.0

        y_values = []

        for column in ["RefPower", "AvgPower"]:
            if column in merged.columns:
                y_values.extend(
                    pd.to_numeric(
                        merged[column],
                        errors="coerce"
                    ).dropna().tolist()
                )

        if len(df_t):
            y_values.extend(
                pd.to_numeric(
                    df_t[power_col],
                    errors="coerce"
                ).dropna().tolist()
            )

        if y_values:
            y_max = float(max(y_values))
            y_upper = max(1000.0, y_max * 1.10)
        else:
            y_upper = 3500.0

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        graph_title = (
            f"{title}  |  Dev: {dev:.2f}%"
            if show_deviation and dev is not None and not pd.isna(dev)
            else title
        )

        fig.update_layout(
            title=dict(
                text=graph_title,
                font=dict(
                    size=18,
                    color=title_color
                ),
                x=0.02,
                xanchor="left"
            ),

            xaxis=dict(
                title=dict(
                    text="Wind Speed (m/s)",
                    font=dict(size=13)
                ),
                tickfont=dict(size=11),
                range=[x_left, x_right],
                showgrid=True,
                zeroline=False,
                automargin=True
            ),

            yaxis=dict(
                title=dict(
                    text="Power (kW)",
                    font=dict(size=13)
                ),
                tickfont=dict(size=11),
                range=[0, y_upper],
                showgrid=True,
                zeroline=False,
                automargin=True
            ),

            height=400,

            margin=dict(
                l=65,
                r=20,
                t=65,
                b=85
            ),

            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.20,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),

            hovermode="x unified",

            plot_bgcolor="white",

            paper_bgcolor="white"
        )

        return fig


    # ======================================================
    # COMMENT
    # ======================================================

    def generate_comment(dev):

        if dev is None or pd.isna(dev):
            return "Deviation could not be calculated."

        dev = round(float(dev), 2)

        if dev < -72:
            return (
                f"Dev: {dev}% → Extreme issue "
                "(data may be unreliable)"
            )

        elif dev < -10:
            return (
                f"Dev: {dev}% → Severe underperformance "
                "(possible Blade/Dust/Yaw issue)"
            )

        elif dev < -2:
            return (
                f"Dev: {dev}% → Underperformance "
                "(possible Control/Availability issue)"
            )

        elif dev > 72:
            return (
                f"Dev: {dev}% → Abnormally high "
                "(possible Sensor/Data issue)"
            )

        elif dev > 8:
            return (
                f"Dev: {dev}% → High overperformance"
            )

        elif dev > 2:
            return (
                f"Dev: {dev}% → Slight overperformance"
            )

        else:
            return (
                f"Dev: {dev}% → Normal performance"
            )

    def status_from_deviation(dev):

        if dev is None or pd.isna(dev):
            return "No Curve"

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

    # ======================================================
    # MODE
    # ======================================================

    # IMPORTANT:
    # Use the COMPLETE turbine list from the original SCADA file.
    # This list is independent of:
    #   - selected date range
    #   - wind/power/pitch filters
    #   - minimum 30-row requirement
    turbines = sorted(
        all_turbines,
        key=natural_sort_key
    )

    st.sidebar.caption(
        f"{len(turbines)} turbines detected in uploaded SCADA"
    )

    if mode == "Single Turbine":

        selected_turbine = st.sidebar.selectbox(
            "Select Turbine",
            turbines,
            key="single_turbine"
        )

        turbines_to_show = [selected_turbine]

    elif mode == "Compare Turbines":

        turbines_to_show = st.sidebar.multiselect(
            "Select Turbines",
            turbines,
            key="compare_turbines"
        )

    else:

        turbines_to_show = turbines

    # ======================================================
    # PROCESS ALL SELECTED TURBINES
    # ======================================================

    processed = []
    results = []

    for t in turbines_to_show:

        item = process_turbine(t)

        dev = item["dev"]
        status = item["status"]

        if item["status"] == "Calculated":

            status = status_from_deviation(dev)
            comment = generate_comment(dev)

        else:

            comment = item["reason"]

        item["status"] = status
        item["comment"] = comment

        processed.append(item)

        results.append({
            "Turbine": t,
            "Valid Rows": item["valid_rows"],
            "Deviation_%": (
                round(float(dev), 2)
                if dev is not None and not pd.isna(dev)
                else np.nan
            ),
            "Status": status
        })

    # ======================================================
    # SUMMARY CARDS
    # ======================================================

    calculated_count = sum(
        1 for x in processed
        if x["status"] in [
            "Normal",
            "Slight Over",
            "High Over",
            "Under",
            "High Under",
            "Issue"
        ]
    )

    insufficient_count = sum(
        1 for x in processed
        if x["status"] == "Insufficient Data"
    )

    no_data_count = sum(
        1 for x in processed
        if x["status"] == "No Data"
    )

    no_curve_count = sum(
        1 for x in processed
        if x["status"] == "No Curve"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Turbines",
            len(turbines_to_show)
        )

    with c2:
        st.metric(
            "Power Curves",
            calculated_count
        )

    with c3:
        st.metric(
            "Insufficient Data",
            insufficient_count
        )

    with c4:
        st.metric(
            "No Data / Curve",
            no_data_count + no_curve_count
        )

    # ======================================================
    # DASHBOARD: 6 READABLE GRAPHS PER PAGE
    # 2 COLUMNS x 3 ROWS
    # ======================================================

    st.subheader("Power Curve Graphs")

    if not processed:

        st.info(
            "No turbines selected. Please select at least one turbine."
        )

    else:

        GRAPHS_PER_PAGE = 6

        total_pages = max(
            1,
            int(
                np.ceil(
                    len(processed) /
                    GRAPHS_PER_PAGE
                )
            )
        )

        if total_pages > 1:

            page = st.sidebar.number_input(
                "Graph Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key=f"graph_page_{site}_{mode}"
            )

        else:

            page = 1

        page = int(page)

        start_idx = (
            page - 1
        ) * GRAPHS_PER_PAGE

        page_items = processed[
            start_idx:
            start_idx + GRAPHS_PER_PAGE
        ]

        end_idx = (
            start_idx +
            len(page_items)
        )

        st.caption(
            f"Showing turbines {start_idx + 1}–{end_idx} "
            f"of {len(processed)}  |  "
            f"Page {page} of {total_pages}  |  "
            f"6 graphs per page"
        )

        # Two columns makes every graph substantially wider
        # and easier to read than 3 narrow columns.
        for row_start in range(
            0,
            len(page_items),
            2
        ):

            row_items = page_items[
                row_start:
                row_start + 2
            ]

            cols = st.columns(
                2,
                gap="large"
            )

            for col, item in zip(
                cols,
                row_items
            ):

                t = item["turbine"]
                df_t = item["df_t"]
                merged = item["merged"]
                dev = item["dev"]
                comment = item["comment"]

                safe_key = re.sub(
                    r"[^a-zA-Z0-9_]",
                    "_",
                    str(t)
                )

                detail_key = (
                    f"show_details_{site}_{safe_key}"
                )

                with col:

                    with st.container(
                        border=True
                    ):

                        show_details = st.checkbox(
                            "Show deviation & comment",
                            value=st.session_state.get(
                                detail_key,
                                True
                            ),
                            key=detail_key
                        )

                        if (
                            merged is not None
                            and
                            not merged.empty
                            and
                            len(df_t) > 0
                        ):

                            fig = plot_graph(
                                df_t,
                                merged,
                                t,
                                dev,
                                show_deviation=show_details
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                config={
                                    "displaylogo": False,
                                    "responsive": True,
                                    "scrollZoom": True
                                },
                                key=(
                                    f"graph_{site}_{safe_key}_{page}"
                                )
                            )

                            if show_details:

                                st.markdown(
                                    f"**Analysis:** {comment}"
                                )

                                st.caption(
                                    f"Valid SCADA rows: "
                                    f"{item['valid_rows']:,} | "
                                    f"Standard deviation: "
                                    f"{item['std']:.2f} kW"
                                    if pd.notna(item["std"])
                                    else
                                    f"Valid SCADA rows: "
                                    f"{item['valid_rows']:,}"
                                )

                        else:

                            st.markdown(
                                f"### {t}"
                            )

                            if item["status"] == "No Data":

                                st.warning(
                                    "No SCADA data is available "
                                    "for this turbine in the selected "
                                    "date range."
                                )

                            elif item["status"] == "Insufficient Data":

                                st.warning(
                                    "Insufficient data for power-curve calculation."
                                )

                            else:

                                st.warning(
                                    "Power curve could not be calculated."
                                )

                            if show_details:

                                st.markdown(
                                    f"**Analysis:** {comment}"
                                )

                            st.caption(
                                f"Raw rows: {item['raw_rows']:,} | "
                                f"Valid rows: {item['valid_rows']:,}"
                            )

    # ======================================================
    # RANKING TABLE DATA
    # ======================================================


    results_df = pd.DataFrame(results)

    # ======================================================
    # RANKING TABLE
    # ======================================================

    st.subheader("Turbine Ranking / Data Status")

    results_df = pd.DataFrame(results)

    if not results_df.empty:

        results_df["_TurbineSort"] = results_df[
            "Turbine"
        ].map(natural_sort_key)

        results_df = (
            results_df
            .sort_values(
                by=["Deviation_%", "_TurbineSort"],
                na_position="last"
            )
            .drop(columns=["_TurbineSort"])
        )

        def color_row(row):

            status = row["Status"]

            if status == "Normal":
                color = "#ccffcc"
            elif status == "Slight Over":
                color = "#b6f5b6"
            elif status == "High Over":
                color = "#8fe08f"
            elif status == "Under":
                color = "#ffd58a"
            elif status == "High Under":
                color = "#ff9999"
            elif status in ["Insufficient Data", "No Data", "No Curve"]:
                color = "#e6e6e6"
            else:
                color = "#f0f0f0"

            return [
                f"background-color: {color}"
            ] * len(row)

        styled_table = (
            results_df.style
            .apply(
                color_row,
                axis=1
            )
            .format(
                {
                    "Deviation_%": lambda x:
                    "" if pd.isna(x)
                    else f"{x:.2f}%"
                }
            )
        )

        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True
        )

    # ======================================================
    # PDF REPORT
    # ======================================================

    try:

        pdf_buffer = io.BytesIO()

        pdf = canvas.Canvas(
            pdf_buffer,
            pagesize=landscape(A4)
        )

        width, height = landscape(A4)

        include_pdf_details = st.checkbox(
            "Include deviation and comments in PDF",
            value=True,
            key="include_pdf_details"
        )

        # Only calculated curves can be rendered as graph images.
        pdf_items = [
            item for item in processed
            if (
                item["merged"] is not None
                and not item["merged"].empty
                and len(item["df_t"]) > 0
                and item["dev"] is not None
                and not pd.isna(item["dev"])
            )
        ]

        if not KALEIDO_AVAILABLE:

            st.warning(
                "PDF can be downloaded, but graph images require "
                "Kaleido. Add `kaleido` to requirements.txt."
            )

        # --------------------------------------------------
        # PDF GRAPH PAGES: 2 COLUMNS x 3 ROWS = 6
        # --------------------------------------------------

        graph_width = 365
        graph_height = 175

        x_positions = [25, 410]
        y_positions = [
            height - 270,
            height - 450,
            height - 630
        ]

        for idx, item in enumerate(pdf_items):

            slot = idx % 6

            if slot == 0:

                if idx != 0:
                    pdf.showPage()

                if os.path.exists(logo_path):

                    pdf.drawImage(
                        logo_path,
                        20,
                        height - 55,
                        width=90,
                        height=30,
                        preserveAspectRatio=True,
                        mask="auto"
                    )

                pdf.setFont(
                    "Helvetica-Bold",
                    13
                )

                pdf.drawString(
                    125,
                    height - 35,
                    "Power Curve Analytics Report"
                )

                pdf.setFont(
                    "Helvetica",
                    8
                )

                pdf.drawString(
                    125,
                    height - 50,
                    f"Site: {site} | "
                    f"{start_day} to {end_day}"
                )

            if not KALEIDO_AVAILABLE:
                continue

            try:

                pdf_fig = plot_graph(
                    item["df_t"],
                    item["merged"],
                    item["turbine"],
                    item["dev"],
                    show_deviation=include_pdf_details
                )

                img = pdf_fig.to_image(
                    format="png"
                )

                img_reader = ImageReader(
                    io.BytesIO(img)
                )

                col = slot % 2
                row = slot // 2

                x = x_positions[col]
                y = y_positions[row]

                pdf.drawImage(
                    img_reader,
                    x,
                    y,
                    width=graph_width,
                    height=graph_height,
                    preserveAspectRatio=True,
                    anchor="c"
                )

                if include_pdf_details:

                    pdf.setFont(
                        "Helvetica-Bold",
                        8
                    )

                    pdf.drawString(
                        x,
                        y - 10,
                        str(item["turbine"])[:45]
                    )

                    pdf.setFont(
                        "Helvetica",
                        7
                    )

                    pdf.drawString(
                        x,
                        y - 20,
                        str(item["comment"])[:90]
                    )

            except Exception:
                # Keep the PDF usable even if one Plotly graph
                # cannot be converted to an image.
                pass

        # --------------------------------------------------
        # DATA STATUS / RANKING SUMMARY
        # --------------------------------------------------

        pdf.showPage()

        if os.path.exists(logo_path):

            pdf.drawImage(
                logo_path,
                25,
                height - 65,
                width=90,
                height=30,
                preserveAspectRatio=True,
                mask="auto"
            )

        pdf.setFont(
            "Helvetica-Bold",
            14
        )

        pdf.drawString(
            130,
            height - 40,
            "Turbine Ranking / Data Status"
        )

        pdf.setFont(
            "Helvetica",
            8
        )

        pdf.drawString(
            130,
            height - 55,
            f"Site: {site} | "
            f"Date: {start_day} to {end_day}"
        )

        y = height - 85

        pdf.setFont(
            "Helvetica-Bold",
            8
        )

        pdf.drawString(
            30,
            y,
            "Turbine"
        )

        pdf.drawString(
            180,
            y,
            "Valid Rows"
        )

        pdf.drawString(
            250,
            y,
            "Deviation"
        )

        pdf.drawString(
            330,
            y,
            "Status"
        )

        y -= 16

        pdf.setFont(
            "Helvetica",
            8
        )

        for _, row in results_df.iterrows():

            deviation = row["Deviation_%"]

            deviation_text = (
                ""
                if pd.isna(deviation)
                else f"{deviation:.2f}%"
            )

            line_values = [
                str(row["Turbine"])[:30],
                str(row["Valid Rows"]),
                deviation_text,
                str(row["Status"])[:25]
            ]

            pdf.drawString(
                30,
                y,
                line_values[0]
            )

            pdf.drawString(
                180,
                y,
                line_values[1]
            )

            pdf.drawString(
                250,
                y,
                line_values[2]
            )

            pdf.drawString(
                330,
                y,
                line_values[3]
            )

            y -= 14

            if y < 35:

                pdf.showPage()

                y = height - 45

                pdf.setFont(
                    "Helvetica",
                    8
                )

        pdf.save()

        pdf_buffer.seek(0)

        st.download_button(
            label="Download Full Dashboard Report (PDF)",
            data=pdf_buffer.getvalue(),
            file_name="WindFarm_Full_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:

        st.error(
            "PDF generation failed"
        )

        st.code(
            str(e)
        )


