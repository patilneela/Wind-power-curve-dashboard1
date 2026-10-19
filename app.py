from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Optional PDF export support
try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


st.set_page_config(page_title="Power Curve Analytics Dashboard", layout="wide")

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
# CONSTANTS
# =========================
WTG_COL = "Name"
TIME_COL = "Time"
STATUS_COL = "PCurveStsAve"
WIND_SPEED_COL = "WindSpeedAve"
WIND_STD_COL = "WindSpeedStd"
PITCH_STD_COL = "BladePitchStd"
POWER_COL = "ActivePWAve"
BIN_WIDTH = 0.5

REQUIRED_SCADA_COLUMNS = [
    WTG_COL,
    TIME_COL,
    STATUS_COL,
    WIND_SPEED_COL,
    WIND_STD_COL,
    PITCH_STD_COL,
    POWER_COL,
]

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
DEFAULT_CSV_PATH = UPLOADS_DIR / "EN-156-3.3_20260801_20260826_ten_minutes-1792412779244.csv"
DEFAULT_XLSX_PATH = UPLOADS_DIR / "All Sites Specific Power Curve data_ENV1-1792412779249.xlsx"
LOGO_PATH = BASE_DIR / "Envision.png"

# =========================
# HELPERS
# =========================
def ensure_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


@st.cache_data(show_spinner=False)
def load_scada_from_upload(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    if not file_name.lower().endswith(".csv"):
        raise ValueError("SCADA input must be a CSV file.")

    scada = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    ensure_required_columns(scada, REQUIRED_SCADA_COLUMNS, "SCADA file")

    for col in [STATUS_COL, WIND_STD_COL, PITCH_STD_COL, WIND_SPEED_COL, POWER_COL]:
        scada[col] = pd.to_numeric(scada[col], errors="coerce")

    scada[TIME_COL] = pd.to_datetime(scada[TIME_COL], errors="coerce", dayfirst=True)
    scada[WTG_COL] = scada[WTG_COL].astype(str).str.strip()
    return scada


@st.cache_data(show_spinner=False)
def load_reference_curves_from_upload(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    if not file_name.lower().endswith(".xlsx"):
        raise ValueError("Reference input must be an XLSX file.")

    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Power Curve", header=None)
    if raw.shape[0] < 4:
        raise ValueError("Reference workbook Power Curve sheet does not contain enough rows.")

    site_row = raw.iloc[2]
    wind_speed = pd.to_numeric(raw.iloc[3:, 0], errors="coerce")
    curves: dict[str, pd.DataFrame] = {}

    for idx in range(1, raw.shape[1]):
        curve_name = site_row.iloc[idx]
        if pd.isna(curve_name):
            continue

        curve_name = str(curve_name).replace("\n", " ").strip()
        if not curve_name:
            continue

        power = pd.to_numeric(raw.iloc[3:, idx], errors="coerce")
        curve_df = pd.DataFrame(
            {
                "ReferenceWindSpeed_mps": wind_speed,
                "ReferencePower_kW": power,
            }
        ).dropna()

        curve_df = (
            curve_df.sort_values("ReferenceWindSpeed_mps")
            .drop_duplicates("ReferenceWindSpeed_mps")
            .reset_index(drop=True)
        )

        if not curve_df.empty:
            curves[curve_name] = curve_df

    if not curves:
        raise ValueError("No usable reference curves were found in the workbook.")

    return curves


def run_analysis(scada: pd.DataFrame, ref: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    working = scada.copy()

    mask_status = working[STATUS_COL].eq(100)
    mask_wind_std = working[WIND_STD_COL].lt(1)
    mask_pitch_std = working[PITCH_STD_COL].between(-5, 19, inclusive="both")
    mask = mask_status & mask_wind_std & mask_pitch_std
    filtered = working.loc[mask].copy()

    filter_counts = pd.DataFrame(
        [
            ["Total SCADA records", len(working)],
            ["PCurveStsAve = 100", int(mask_status.sum())],
            [
                "After PCurveStsAve = 100 and WindSpeedStd < 1",
                int((mask_status & mask_wind_std).sum()),
            ],
            ["After all three required filters", int(mask.sum())],
        ],
        columns=["Filter Stage", "Record Count"],
    )

    filtered = filtered.rename(
        columns={
            WTG_COL: "WTG",
            TIME_COL: "DateTime",
            WIND_SPEED_COL: "WindSpeed_mps",
            POWER_COL: "ActualPower_kW",
        }
    )

    ref_working = ref.copy()
    for col in ["ReferenceWindSpeed_mps", "ReferencePower_kW"]:
        ref_working[col] = pd.to_numeric(ref_working[col], errors="coerce")

    ref_working = (
        ref_working.dropna()
        .sort_values("ReferenceWindSpeed_mps")
        .drop_duplicates("ReferenceWindSpeed_mps")
        .reset_index(drop=True)
    )

    if ref_working.empty:
        raise ValueError("Selected reference curve has no valid numeric data.")

    ref_min = ref_working["ReferenceWindSpeed_mps"].min()
    ref_max = ref_working["ReferenceWindSpeed_mps"].max()

    filtered = filtered[filtered["WindSpeed_mps"].between(ref_min, ref_max)].copy()
    if filtered.empty:
        raise ValueError("No SCADA records remain after applying filters and reference wind-speed range.")

    filtered["ReferencePower_kW"] = np.interp(
        filtered["WindSpeed_mps"],
        ref_working["ReferenceWindSpeed_mps"],
        ref_working["ReferencePower_kW"],
    )
    filtered["Deviation_kW"] = filtered["ActualPower_kW"] - filtered["ReferencePower_kW"]
    filtered["Shortfall_kW"] = -filtered["Deviation_kW"]
    filtered["Deviation_pct"] = np.where(
        filtered["ReferencePower_kW"] > 0,
        100 * filtered["Deviation_kW"] / filtered["ReferencePower_kW"],
        np.nan,
    )
    filtered["WindSpeedBin_mps"] = np.floor(filtered["WindSpeed_mps"] / BIN_WIDTH) * BIN_WIDTH

    wtg_bins = (
        filtered.groupby(["WTG", "WindSpeedBin_mps"])
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
        .reset_index()
    )

    fleet_bins = (
        filtered.groupby("WindSpeedBin_mps")
        .agg(
            Samples=("ActualPower_kW", "size"),
            MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
            MeanActualPower_kW=("ActualPower_kW", "mean"),
            MeanReferencePower_kW=("ReferencePower_kW", "mean"),
            MeanDeviation_kW=("Deviation_kW", "mean"),
            MeanDeviation_pct=("Deviation_pct", "mean"),
        )
        .reset_index()
    )

    wtg_summary = (
        filtered.groupby("WTG")
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
        .reset_index()
    )

    wtg_summary["AggregateDeviation_pct"] = np.where(
        wtg_summary["MeanReferencePower_kW"] > 0,
        100
        * (wtg_summary["MeanActualPower_kW"] - wtg_summary["MeanReferencePower_kW"])
        / wtg_summary["MeanReferencePower_kW"],
        np.nan,
    )
    wtg_summary = wtg_summary.sort_values("AggregateDeviation_pct")

    overall = pd.DataFrame(
        [
            ["Total SCADA records", len(working)],
            ["Records passing all required filters", int(mask.sum())],
            ["Records compared to reference curve", len(filtered)],
            ["WTGs compared", filtered["WTG"].nunique()],
            ["Reference curve min wind speed (m/s)", ref_min],
            ["Reference curve max wind speed (m/s)", ref_max],
            ["Filter 1", "PCurveStsAve = 100"],
            ["Filter 2", "WindSpeedStd < 1"],
            ["Filter 3", "-5 <= BladePitchStd <= 19"],
            [
                "Deviation sign",
                "Actual power - reference power; negative means below reference",
            ],
        ],
        columns=["Metric", "Value"],
    )

    return {
        "filtered": filtered,
        "filter_counts": filter_counts,
        "wtg_bins": wtg_bins,
        "fleet_bins": fleet_bins,
        "wtg_summary": wtg_summary,
        "overall": overall,
        "ref": ref_working,
        "rows_passing": int(mask.sum()),
        "rows_compared": len(filtered),
        "wtg_count": int(filtered["WTG"].nunique()),
    }


def make_fleet_power_curve_plotly(ref: pd.DataFrame, filtered: pd.DataFrame, fleet_bins: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=filtered["WindSpeed_mps"],
            y=filtered["ActualPower_kW"],
            mode="markers",
            name="Filtered 10-min SCADA",
            marker=dict(size=5, opacity=0.18, color="#1f77b4"),
            hovertemplate="Wind Speed: %{x:.2f} m/s<br>Actual Power: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ref["ReferenceWindSpeed_mps"],
            y=ref["ReferencePower_kW"],
            mode="lines",
            name="Reference Power Curve",
            line=dict(width=3, color="#d62728"),
            hovertemplate="Reference Wind: %{x:.2f} m/s<br>Reference Power: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fleet_bins["MeanWindSpeed_mps"],
            y=fleet_bins["MeanActualPower_kW"],
            mode="lines+markers",
            name="SCADA mean by 0.5 m/s bin",
            line=dict(width=3, color="#2ca02c"),
            marker=dict(size=7),
            hovertemplate="Mean Wind: %{x:.2f} m/s<br>Mean Actual Power: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.update_layout(
        title="Fleet Power Curve Comparison",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=650,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def make_fleet_deviation_plotly(fleet_bins: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fleet_bins["MeanWindSpeed_mps"],
            y=fleet_bins["MeanDeviation_kW"],
            mode="lines+markers",
            name="Mean Deviation",
            line=dict(width=3, color="#ff7f0e"),
            marker=dict(size=7),
            hovertemplate="Mean Wind: %{x:.2f} m/s<br>Mean Deviation: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.add_hline(y=0, line_width=1, line_color="black")

    fig.update_layout(
        title="Fleet Deviation by Wind-Speed Bin",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Mean Deviation (Actual - Reference, kW)",
        height=500,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def make_wtg_power_curve_plotly(ref: pd.DataFrame, filtered_wtg: pd.DataFrame, wtg_bins: pd.DataFrame, wtg_name: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=filtered_wtg["WindSpeed_mps"],
            y=filtered_wtg["ActualPower_kW"],
            mode="markers",
            name=f"{wtg_name} SCADA",
            marker=dict(size=6, opacity=0.20, color="#1f77b4"),
            hovertemplate="Wind Speed: %{x:.2f} m/s<br>Actual Power: %{y:.2f} kW<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ref["ReferenceWindSpeed_mps"],
            y=ref["ReferencePower_kW"],
            mode="lines",
            name="Reference Curve",
            line=dict(width=3, color="#d62728"),
        )
    )

    if not wtg_bins.empty:
        fig.add_trace(
            go.Scatter(
                x=wtg_bins["MeanWindSpeed_mps"],
                y=wtg_bins["MeanActualPower_kW"],
                mode="lines+markers",
                name=f"{wtg_name} mean by bin",
                line=dict(width=3, color="#2ca02c"),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title=f"{wtg_name} Power Curve Comparison",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def make_multi_wtg_comparison_plot(results: dict, selected_wtgs: list[str]):
    fig = go.Figure()

    subset = results["wtg_bins"][results["wtg_bins"]["WTG"].isin(selected_wtgs)].copy()
    for wtg in selected_wtgs:
        s = subset[subset["WTG"] == wtg].copy()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s["MeanWindSpeed_mps"],
                y=s["MeanActualPower_kW"],
                mode="lines+markers",
                name=wtg,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=results["ref"]["ReferenceWindSpeed_mps"],
            y=results["ref"]["ReferencePower_kW"],
            mode="lines",
            name="Reference Curve",
            line=dict(width=4, color="black", dash="dash"),
        )
    )

    fig.update_layout(
        title="Multi-WTG Comparison vs Reference Curve",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Mean Actual Power (kW)",
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_report(site_name: str, date_text: str, figures: list[tuple[str, go.Figure]], summary_df: pd.DataFrame):
    if not (KALEIDO_AVAILABLE and REPORTLAB_AVAILABLE):
        raise RuntimeError("PDF export needs both kaleido and reportlab installed.")

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=landscape(A4))
    width, height = landscape(A4)

    if LOGO_PATH.exists():
        pdf.drawImage(str(LOGO_PATH), 30, height - 80, width=120, height=40, preserveAspectRatio=True, mask="auto")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(170, height - 40, "Power Curve Analytics Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(170, height - 60, f"Reference Curve: {site_name}")
    pdf.drawString(170, height - 75, f"Date Range: {date_text}")

    y = height - 120

    for title, fig in figures:
        try:
            img = fig.to_image(format="png", width=1200, height=700, scale=2)
            img_reader = ImageReader(io.BytesIO(img))

            if y < 260:
                pdf.showPage()
                y = height - 60

            pdf.setFont("Helvetica-Bold", 11)
            pdf.drawString(30, y, title)
            pdf.drawImage(img_reader, 30, y - 220, width=500, height=200)
            y -= 250
        except Exception:
            continue

    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(30, height - 40, "WTG Ranking Summary")
    y = height - 80
    pdf.setFont("Helvetica", 10)

    for _, row in summary_df.iterrows():
        line = (
            f"{row['WTG']} | AggregateDeviation_pct: "
            f"{'NA' if pd.isna(row['AggregateDeviation_pct']) else round(row['AggregateDeviation_pct'], 2)} | "
            f"ValidSamples: {row['ValidSamples']}"
        )
        pdf.drawString(40, y, line[:170])
        y -= 20
        if y < 40:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 40

    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# =========================
# HEADER
# =========================
top_left, top_mid, top_right = st.columns([1, 2, 1])
with top_mid:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=260)

st.title("Power Curve Analytics Dashboard")
st.caption("Correct engineering logic with polished UI, interactive charts, optional login, logo, and PDF export")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Inputs")

uploaded_scada = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])
uploaded_ref = st.sidebar.file_uploader("Upload Reference Excel", type=["xlsx"])

use_default_files = st.sidebar.checkbox("Use default local files if uploads are not provided", value=True)

scada_bytes = None
scada_name = None
ref_bytes = None
ref_name = None

if uploaded_scada is not None:
    scada_bytes = uploaded_scada.getvalue()
    scada_name = uploaded_scada.name
elif use_default_files and DEFAULT_CSV_PATH.exists():
    scada_bytes = DEFAULT_CSV_PATH.read_bytes()
    scada_name = DEFAULT_CSV_PATH.name

if uploaded_ref is not None:
    ref_bytes = uploaded_ref.getvalue()
    ref_name = uploaded_ref.name
elif use_default_files and DEFAULT_XLSX_PATH.exists():
    ref_bytes = DEFAULT_XLSX_PATH.read_bytes()
    ref_name = DEFAULT_XLSX_PATH.name

if scada_bytes is None:
    st.warning("Please upload the SCADA CSV file.")
    st.stop()

if ref_bytes is None:
    st.warning("Please upload the reference Excel file.")
    st.stop()

# =========================
# LOAD DATA
# =========================
try:
    scada = load_scada_from_upload(scada_bytes, scada_name)
    reference_curves = load_reference_curves_from_upload(ref_bytes, ref_name)
except Exception as exc:
    st.error(f"Failed to load input files: {exc}")
    st.stop()

# =========================
# CONTROLS
# =========================
st.sidebar.header("Analysis Controls")

curve_names = sorted(reference_curves.keys())
selected_curve = st.sidebar.selectbox("Select reference curve", curve_names)

valid_time = scada[TIME_COL].dropna()
if not valid_time.empty:
    min_date = valid_time.min().date()
    max_date = valid_time.max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    date_range = None

scada_filtered = scada.copy()
date_text = "All available dates"

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    scada_filtered = scada_filtered[
        scada_filtered[TIME_COL].dt.date.between(start_date, end_date)
    ].copy()
    date_text = f"{start_date} to {end_date}"

all_wtgs = sorted(scada_filtered[WTG_COL].dropna().astype(str).unique().tolist())
selected_wtgs_filter = st.sidebar.multiselect(
    "Optional WTG filter",
    options=all_wtgs,
    default=[],
    help="Leave empty to analyse all available WTG rows in the selected date range.",
)

if selected_wtgs_filter:
    scada_filtered = scada_filtered[scada_filtered[WTG_COL].isin(selected_wtgs_filter)].copy()

if scada_filtered.empty:
    st.warning("No SCADA data is available after applying the selected date/WTG filters.")
    st.stop()

# =========================
# RUN ANALYSIS
# =========================
ref_df = reference_curves[selected_curve]

try:
    results = run_analysis(scada_filtered, ref_df)
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.stop()

# =========================
# METRICS
# =========================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows passing filters", f"{results['rows_passing']:,}")
m2.metric("Rows compared to curve", f"{results['rows_compared']:,}")
m3.metric("WTGs analysed", f"{results['wtg_count']:,}")
m4.metric("Reference curve", selected_curve)

# =========================
# SAFE WTG VIEW SELECTION
# =========================
available_wtgs = sorted(results["filtered"]["WTG"].dropna().unique().tolist())

default_wtgs = available_wtgs[: min(8, len(available_wtgs))]
selected_wtgs = st.multiselect(
    "Select turbines to view",
    options=available_wtgs,
    default=default_wtgs,
    help="Default selection is limited for better performance. Add more turbines if needed.",
)

if not selected_wtgs:
    st.warning("Please select at least one turbine to display.")
    st.stop()

filtered_view = results["filtered"][results["filtered"]["WTG"].isin(selected_wtgs)].copy()
wtg_summary_view = results["wtg_summary"][results["wtg_summary"]["WTG"].isin(selected_wtgs)].copy()
wtg_bins_view = results["wtg_bins"][results["wtg_bins"]["WTG"].isin(selected_wtgs)].copy()

fleet_bins_view = (
    filtered_view.groupby("WindSpeedBin_mps")
    .agg(
        Samples=("ActualPower_kW", "size"),
        MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
        MeanActualPower_kW=("ActualPower_kW", "mean"),
        MeanReferencePower_kW=("ReferencePower_kW", "mean"),
        MeanDeviation_kW=("Deviation_kW", "mean"),
        MeanDeviation_pct=("Deviation_pct", "mean"),
    )
    .reset_index()
)

# =========================
# SUMMARY TABLES
# =========================
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Filter counts")
    st.dataframe(results["filter_counts"], width="stretch", hide_index=True)

with right_col:
    st.subheader("Overall summary")
    overall_display = results["overall"].astype({"Value": "string"})
    st.dataframe(overall_display, width="stretch", hide_index=True)

# =========================
# FLEET CHARTS
# =========================
st.subheader("Fleet charts")

fleet_power_fig = make_fleet_power_curve_plotly(
    results["ref"], filtered_view, fleet_bins_view
)
st.plotly_chart(fleet_power_fig, width="stretch")

fleet_dev_fig = make_fleet_deviation_plotly(fleet_bins_view)
st.plotly_chart(fleet_dev_fig, width="stretch")

# =========================
# SELECTED WTG SUMMARY
# =========================
st.subheader("Selected WTG summary")
st.dataframe(wtg_summary_view, width="stretch", hide_index=True)

# =========================
# DATA TABLES
# =========================
with st.expander("Show selected turbines wind bin summary"):
    st.dataframe(fleet_bins_view, width="stretch", hide_index=True)

with st.expander("Show selected WTG wind bin summary"):
    st.dataframe(wtg_bins_view, width="stretch", hide_index=True)

with st.expander("Show filtered SCADA sample"):
    preview_cols = [
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
    st.dataframe(
        filtered_view[preview_cols].head(200),
        width="stretch",
        hide_index=True,
    )

# =========================
# DOWNLOADS
# =========================
st.subheader("Downloads")

d1, d2, d3 = st.columns(3)

with d1:
    st.download_button(
        "Download filtered SCADA CSV",
        data=to_csv_bytes(filtered_view),
        file_name="filtered_scada_power_curve.csv",
        mime="text/csv",
    )

with d2:
    st.download_button(
        "Download WTG summary CSV",
        data=to_csv_bytes(wtg_summary_view),
        file_name="wtg_summary_power_curve.csv",
        mime="text/csv",
    )

with d3:
    st.download_button(
        "Download selected turbines bins CSV",
        data=to_csv_bytes(fleet_bins_view),
        file_name="selected_turbines_bins_power_curve.csv",
        mime="text/csv",
    )

# =========================
# PDF EXPORT
# =========================
st.subheader("PDF report")

if KALEIDO_AVAILABLE and REPORTLAB_AVAILABLE:
    pdf_figures = [
        ("Fleet Power Curve Comparison", fleet_power_fig),
        ("Fleet Deviation by Wind-Speed Bin", fleet_dev_fig),
    ]

    try:
        pdf_bytes = build_pdf_report(
            selected_curve,
            date_text,
            pdf_figures,
            wtg_summary_view,
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="power_curve_dashboard_report.pdf",
            mime="application/pdf",
        )
    except Exception as exc:
        st.warning(f"PDF generation is available but failed: {exc}")
else:
    st.info("PDF export is disabled. Install both kaleido and reportlab to enable PDF download.")
