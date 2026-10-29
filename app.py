from pathlib import Path
import io
import zipfile
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
    KALEIDO_IMPORT_ERROR = ""
except Exception as exc:
    KALEIDO_AVAILABLE = False
    KALEIDO_IMPORT_ERROR = str(exc)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
    REPORTLAB_IMPORT_ERROR = ""
except Exception as exc:
    REPORTLAB_AVAILABLE = False
    REPORTLAB_IMPORT_ERROR = str(exc)

st.set_page_config(page_title="Power Curve Analytic Report", layout="wide")

WTG_COL = "Name"
TIME_COL = "Time"
STATUS_COL = "PCurveStsAve"
WIND_SPEED_COL = "WindSpeedAve"
WIND_STD_COL = "WindSpeedStd"
PITCH_STD_COL = "BladePitchStd"
POWER_COL = "ActivePWAve"
NACELLE_COL = "NacellePositionAve"
NACELLE_STD_COL = "NacellePositionStd"
BIN_WIDTH = 0.5
LOW_DATA_THRESHOLD = 30

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
LOGO_PATH = BASE_DIR / "Envision.png"


def ensure_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def get_status_and_comment(perf):
    if perf is None or pd.isna(perf):
        return "NA", "Performance not available"

    perf = round(perf, 2)

    if perf < 28:
        return "Extreme Issue", "Extreme issue (Data unreliable)"
    elif perf < 92:
        return "Severe Underperformance", "Severe underperformance (Blade/Dust/Yaw issue)"
    elif perf < 98:
        return "Underperformance", "Underperformance"
    elif perf > 172:
        return "Abnormal High", "Abnormal high (Sensor/Data issue)"
    elif perf > 108:
        return "High Overperformance", "High overperformance"
    elif perf > 102:
        return "Slight Overperformance", "Slight overperformance"
    else:
        return "Normal", "Normal performance"


def get_status_color(status: str) -> str:
    color_map = {
        "Extreme Issue": "#8B0000",
        "Severe Underperformance": "#E53935",
        "Underperformance": "#FB8C00",
        "Normal": "#43A047",
        "Slight Overperformance": "#1E88E5",
        "High Overperformance": "#3949AB",
        "Abnormal High": "#6A1B9A",
        "NA": "#9E9E9E",
    }
    return color_map.get(status, "#9E9E9E")


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


def get_nacelle_band_color(band: str) -> str:
    color_map = {
        "0-60": "#2E7D32",
        "60-120": "#FDD835",
        "120-180": "#FB8C00",
        "180-240": "#EC407A",
        "240-360": "#E53935",
        "Out of Range": "#6D4C41",
        "NA": "#9E9E9E",
    }
    return color_map.get(band, "#9E9E9E")


def hex_to_reportlab_color(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return colors.HexColor(f"#{hex_color}")


def style_wtg_summary(df: pd.DataFrame):
    def style_status(val):
        color = get_status_color(val)
        return f"background-color: {color}; color: white; font-weight: bold;"

    def style_performance(val):
        if pd.isna(val):
            return ""
        status, _ = get_status_and_comment(val)
        color = get_status_color(status)
        return f"background-color: {color}; color: white; font-weight: bold;"

    styled = df.style.format(
        {
            "Performance_pct": "{:.2f}",
            "MeanWindSpeed_mps": "{:.2f}",
            "MeanActualPower_kW": "{:.2f}",
            "MeanReferencePower_kW": "{:.2f}",
            "MeanDeviation_kW": "{:.2f}",
            "MeanSampleDeviation_pct": "{:.2f}",
            "AggregateDeviation_pct": "{:.2f}",
        }
    )

    if "Status" in df.columns:
        styled = styled.map(style_status, subset=["Status"])

    if "Performance_pct" in df.columns:
        styled = styled.map(style_performance, subset=["Performance_pct"])

    return styled


def style_nacelle_summary(df: pd.DataFrame):
    def style_band(val):
        color = get_nacelle_band_color(val)
        text_color = "black" if val == "60-120" else "white"
        return f"background-color: {color}; color: {text_color}; font-weight: bold;"

    styled = df.style.format(
        {
            "MeanWindSpeed_mps": "{:.2f}",
            "MeanNacellePosition": "{:.2f}",
        }
    )

    if "NacelleBand" in df.columns:
        styled = styled.map(style_band, subset=["NacelleBand"])

    return styled


def render_status_badge(wtg: str, perf: float, dev: float, comment: str):
    status, _ = get_status_and_comment(perf)
    color = get_status_color(status)

    perf_text = "NA" if pd.isna(perf) else f"{perf:.2f}%"
    dev_text = "NA" if pd.isna(dev) else f"{dev:.2f}%"
    comment_text = comment if comment and str(comment).strip() else "Performance comment not available"

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            color:white;
            padding:14px 18px;
            border-radius:12px;
            margin:8px 0 14px 0;
            font-weight:600;
            box-shadow:0 2px 8px rgba(0,0,0,0.12);
        ">
            <div style="font-size:20px; margin-bottom:6px;">{wtg}</div>
            <div style="font-size:16px; margin-bottom:4px;">
                Performance: {perf_text} | Deviation: {dev_text}
            </div>
            <div style="font-size:14px; line-height:1.5;">
                Comment: {comment_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nacelle_badge(wtg: str, avg_nacelle: float, band: str):
    color = get_nacelle_band_color(band)
    text_color = "black" if band == "60-120" else "white"
    nacelle_text = "NA" if pd.isna(avg_nacelle) else f"{avg_nacelle:.2f}°"

    st.markdown(
        f"""
        <div style="
            background-color:{color};
            color:{text_color};
            padding:14px 18px;
            border-radius:12px;
            margin:8px 0 14px 0;
            font-weight:600;
            box-shadow:0 2px 8px rgba(0,0,0,0.12);
        ">
            <div style="font-size:20px; margin-bottom:6px;">{wtg}</div>
            <div style="font-size:16px; margin-bottom:4px;">
                Avg Nacelle Position: {nacelle_text}
            </div>
            <div style="font-size:14px; line-height:1.5;">
                Band: {band}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_last_month_range(today_value: date) -> tuple[date, date]:
    first_day_this_month = today_value.replace(day=1)
    last_day_last_month = first_day_this_month - timedelta(days=1)
    first_day_last_month = last_day_last_month.replace(day=1)
    return first_day_last_month, last_day_last_month


def draw_pdf_header(pdf, width, height, title, site_name, date_text):
    if LOGO_PATH.exists():
        pdf.drawImage(str(LOGO_PATH), 30, height - 70, width=120, height=35, preserveAspectRatio=True, mask="auto")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(170, height - 40, title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(170, height - 58, f"Site Name: {site_name}")
    pdf.drawString(170, height - 72, f"Date Range: {date_text}")
    pdf.drawString(170, height - 86, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def draw_badge_pdf(pdf, x, y, width, height, fill_color_hex, title_text, line1, line2=None, text_color_hex="#FFFFFF"):
    fill_color = hex_to_reportlab_color(fill_color_hex)
    text_color = hex_to_reportlab_color(text_color_hex)

    pdf.setFillColor(fill_color)
    pdf.roundRect(x, y, width, height, 10, fill=1, stroke=0)

    pdf.setFillColor(text_color)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(x + 12, y + height - 18, title_text[:90])

    pdf.setFont("Helvetica", 10)
    pdf.drawString(x + 12, y + height - 34, line1[:140])

    if line2:
        pdf.drawString(x + 12, y + height - 48, line2[:140])


def draw_simple_table(pdf, x, y_top, col_widths, headers, rows, row_height=18, header_fill="#D9EAF7"):
    page_width, page_height = landscape(A4)
    y = y_top

    pdf.setFillColor(hex_to_reportlab_color(header_fill))
    pdf.rect(x, y - row_height, sum(col_widths), row_height, fill=1, stroke=1)

    current_x = x
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica-Bold", 9)
    for header, width in zip(headers, col_widths):
        pdf.drawString(current_x + 4, y - 12, str(header)[:28])
        current_x += width

    y -= row_height

    pdf.setFont("Helvetica", 8)
    for row in rows:
        if y < 40:
            pdf.showPage()
            y = page_height - 40

        current_x = x
        pdf.setFillColor(colors.black)
        for cell, width in zip(row, col_widths):
            pdf.rect(current_x, y - row_height, width, row_height, fill=0, stroke=1)
            text = "" if pd.isna(cell) else str(cell)
            pdf.drawString(current_x + 4, y - 12, text[:28])
            current_x += width
        y -= row_height

    return y


def build_dashboard_pdf(site_name: str, date_text: str, results: dict) -> bytes:
    if not (KALEIDO_AVAILABLE and REPORTLAB_AVAILABLE):
        raise RuntimeError("PDF export needs both kaleido and reportlab installed.")

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=landscape(A4))
    width, height = landscape(A4)

    # Page 1: Executive summary
    pdf.showPage()
    draw_pdf_header(pdf, width, height, "Power Curve Dashboard Report", site_name, date_text)

    upload_summary = results["upload_summary"]
    coverage_summary = results["turbine_coverage_summary"]

    summary_lines = [
        f"Uploaded files: {upload_summary['uploaded_file_count']}",
        f"Valid SCADA files: {upload_summary['valid_file_count']}",
        f"Invalid files: {upload_summary['invalid_file_count']}",
        f"Rows uploaded: {upload_summary['total_rows_uploaded']:,}",
        f"Rows passing filters: {results['rows_passing']:,}",
        f"Rows compared: {results['rows_compared']:,}",
        f"Uploaded turbines: {coverage_summary['uploaded_turbines']}",
        f"Working turbines: {coverage_summary['working_turbines']}",
        f"Missing turbines: {coverage_summary['missing_turbines']}",
        f"Low-data turbines: {coverage_summary['low_data_turbines']}",
    ]

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, height - 120, "Executive Summary")
    pdf.setFont("Helvetica", 11)

    y = height - 145
    for line in summary_lines:
        pdf.drawString(40, y, line)
        y -= 18

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, y - 10, "Filter Summary")
    y -= 25

    filter_rows = results["filter_counts"].values.tolist()
    y = draw_simple_table(
        pdf,
        x=30,
        y_top=y,
        col_widths=[350, 120],
        headers=["Filter Stage", "Record Count"],
        rows=filter_rows,
        row_height=20,
    )

    # Page 2: File ingest summary
    pdf.showPage()
    draw_pdf_header(pdf, width, height, "Power Curve Dashboard Report", site_name, date_text)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, height - 120, "File Ingest Summary")

    file_rows = results["file_ingest_summary"].copy()
    if not file_rows.empty:
        y = draw_simple_table(
            pdf,
            x=30,
            y_top=height - 140,
            col_widths=[240, 90, 80, 270],
            headers=["File", "Status", "Rows", "Message"],
            rows=file_rows[["FileName", "Status", "RowsLoaded", "Message"]].values.tolist(),
            row_height=20,
        )
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(40, height - 145, "No file ingest summary available.")
        y = height - 165

    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, y - 10, "Turbine Coverage Summary")
    y -= 28

    coverage_rows = [[k.replace("_", " ").title(), v] for k, v in coverage_summary.items()]
    y = draw_simple_table(
        pdf,
        x=30,
        y_top=y,
        col_widths=[260, 120],
        headers=["Metric", "Value"],
        rows=coverage_rows,
        row_height=20,
    )

    # Page 3: Missing turbines
    pdf.showPage()
    draw_pdf_header(pdf, width, height, "Power Curve Dashboard Report", site_name, date_text)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, height - 120, "Missing Turbines")

    missing_df = results["missing_turbines"]
    if not missing_df.empty:
        draw_simple_table(
            pdf,
            x=30,
            y_top=height - 140,
            col_widths=[140, 120, 130, 320],
            headers=["WTG", "Uploaded Rows", "Date Range Rows", "Reason"],
            rows=missing_df[["WTG", "UploadedRows", "RowsInDateRange", "Reason"]].values.tolist(),
            row_height=20,
        )
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(40, height - 145, "No missing turbines.")

    # Page 4: Low-data turbines
    pdf.showPage()
    draw_pdf_header(pdf, width, height, "Power Curve Dashboard Report", site_name, date_text)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, height - 120, "Low-Data Turbines")

    low_df = results["low_data_turbines"]
    if not low_df.empty:
        draw_simple_table(
            pdf,
            x=30,
            y_top=height - 140,
            col_widths=[160, 140, 160, 260],
            headers=["WTG", "Valid Samples", "Threshold", "Reason"],
            rows=low_df[["WTG", "ValidSamples", "Threshold", "Reason"]].values.tolist(),
            row_height=20,
        )
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(40, height - 145, "No low-data turbines.")

    # Page 5: WTG summary
    pdf.showPage()
    draw_pdf_header(pdf, width, height, "Power Curve Dashboard Report", site_name, date_text)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(30, height - 120, "WTG Summary")

    wtg_summary_pdf = results["wtg_summary"].copy()
    if not wtg_summary_pdf.empty:
        cols = ["WTG", "ValidSamples", "Performance_pct", "Status", "AggregateDeviation_pct"]
        draw_simple_table(
            pdf,
            x=30,
            y_top=height - 140,
            col_widths=[140, 100, 110, 180, 130],
            headers=cols,
            rows=wtg_summary_pdf[cols].fillna("").values.tolist(),
            row_height=20,
        )
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(40, height - 145, "No WTG summary available.")

    # Per turbine power curve pages
    summary_lookup = results["wtg_summary"].set_index("WTG") if not results["wtg_summary"].empty else pd.DataFrame()
    all_wtgs = sorted(results["filtered"]["WTG"].dropna().unique().tolist())

    for wtg in all_wtgs:
        filtered_wtg = results["filtered"][results["filtered"]["WTG"] == wtg].copy()
        wtg_bins = results["wtg_bins"][results["wtg_bins"]["WTG"] == wtg].copy()
        if filtered_wtg.empty:
            continue

        perf = summary_lookup.loc[wtg, "Performance_pct"] if wtg in summary_lookup.index else np.nan
        dev = summary_lookup.loc[wtg, "AggregateDeviation_pct"] if wtg in summary_lookup.index else np.nan
        comment = summary_lookup.loc[wtg, "PerformanceComment"] if wtg in summary_lookup.index else "Performance comment not available"
        status = summary_lookup.loc[wtg, "Status"] if wtg in summary_lookup.index else "NA"
        badge_color = get_status_color(status)

        fig = make_wtg_power_curve_plotly(results["ref"], filtered_wtg, wtg_bins, wtg)
        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=1)
        img_reader = ImageReader(io.BytesIO(img_bytes))

        pdf.showPage()
        draw_pdf_header(pdf, width, height, "Power Curve Dashboard Report", site_name, date_text)

        perf_text = "NA" if pd.isna(perf) else f"{perf:.2f}%"
        dev_text = "NA" if pd.isna(dev) else f"{dev:.2f}%"

        draw_badge_pdf(
            pdf,
            x=30,
            y=height - 155,
            width=width - 60,
            height=60,
            fill_color_hex=badge_color,
            title_text=wtg,
            line1=f"Performance: {perf_text} | Deviation: {dev_text}",
            line2=f"Comment: {comment}",
        )

        pdf.drawImage(img_reader, 30, 40, width=width - 60, height=height - 230, preserveAspectRatio=True, mask="auto")

    # Nacelle section
    if results["has_nacelle"] and not results["nacelle_summary"].empty:
        pdf.showPage()
        draw_pdf_header(pdf, width, height, "Nacelle Position Report", site_name, date_text)
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(30, height - 120, "Nacelle Summary")

        nacelle_cols = ["WTG", "ValidSamples", "MeanWindSpeed_mps", "MeanNacellePosition", "NacelleBand"]
        draw_simple_table(
            pdf,
            x=30,
            y_top=height - 140,
            col_widths=[120, 100, 130, 150, 120],
            headers=nacelle_cols,
            rows=results["nacelle_summary"][nacelle_cols].fillna("").values.tolist(),
            row_height=20,
        )

        nacelle_lookup = results["nacelle_summary"].set_index("WTG")
        all_nacelle_wtgs = sorted(results["nacelle_summary"]["WTG"].dropna().unique().tolist())

        for wtg in all_nacelle_wtgs:
            filtered_wtg = results["filtered"][results["filtered"]["WTG"] == wtg].copy()
            filtered_wtg = filtered_wtg.dropna(subset=["NacellePositionAve"]).copy()
            nacelle_bins = results["nacelle_bins"][results["nacelle_bins"]["WTG"] == wtg].copy()

            if filtered_wtg.empty:
                continue

            avg_nacelle = nacelle_lookup.loc[wtg, "MeanNacellePosition"] if wtg in nacelle_lookup.index else np.nan
            band = nacelle_lookup.loc[wtg, "NacelleBand"] if wtg in nacelle_lookup.index else "NA"
            badge_color = get_nacelle_band_color(band)
            text_color = "#000000" if band == "60-120" else "#FFFFFF"

            fig = make_nacelle_plotly(results["ref"], filtered_wtg, nacelle_bins, wtg)
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=1)
            img_reader = ImageReader(io.BytesIO(img_bytes))

            pdf.showPage()
            draw_pdf_header(pdf, width, height, "Nacelle Position Report", site_name, date_text)

            avg_text = "NA" if pd.isna(avg_nacelle) else f"{avg_nacelle:.2f}°"

            draw_badge_pdf(
                pdf,
                x=30,
                y=height - 155,
                width=width - 60,
                height=60,
                fill_color_hex=badge_color,
                title_text=wtg,
                line1=f"Avg Nacelle Position: {avg_text}",
                line2=f"Band: {band}",
                text_color_hex=text_color,
            )

            pdf.drawImage(img_reader, 30, 40, width=width - 60, height=height - 230, preserveAspectRatio=True, mask="auto")

    # Comparison section
    comparison_wtgs = sorted(results["filtered"]["WTG"].dropna().unique().tolist())
    if comparison_wtgs:
        selected_for_report = comparison_wtgs[: min(10, len(comparison_wtgs))]
        comparison_filtered = results["filtered"][results["filtered"]["WTG"].isin(selected_for_report)].copy()
        comparison_bins = results["wtg_bins"][results["wtg_bins"]["WTG"].isin(selected_for_report)].copy()
        comparison_fig = make_multi_turbine_comparison_plotly(
            results["ref"],
            comparison_filtered,
            comparison_bins,
            selected_for_report,
            show_scatter=False,
        )
        comparison_img_bytes = comparison_fig.to_image(format="png", width=1200, height=700, scale=1)
        comparison_img_reader = ImageReader(io.BytesIO(comparison_img_bytes))

        pdf.showPage()
        draw_pdf_header(pdf, width, height, "Multi-Turbine Comparison", site_name, date_text)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(30, height - 110, f"Turbines included in report comparison: {', '.join(selected_for_report[:10])}")
        pdf.drawImage(comparison_img_reader, 30, 40, width=width - 60, height=height - 180, preserveAspectRatio=True, mask="auto")

    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_single_scada_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    if not file_name.lower().endswith(".csv"):
        raise ValueError("SCADA input must be a CSV file.")

    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    ensure_required_columns(df, REQUIRED_SCADA_COLUMNS, f"SCADA file '{file_name}'")

    numeric_cols = [STATUS_COL, WIND_STD_COL, PITCH_STD_COL, WIND_SPEED_COL, POWER_COL]
    if NACELLE_COL in df.columns:
        numeric_cols.append(NACELLE_COL)
    if NACELLE_STD_COL in df.columns:
        numeric_cols.append(NACELLE_STD_COL)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", dayfirst=True)
    df[WTG_COL] = df[WTG_COL].astype(str).str.strip()
    df["SourceFile"] = file_name
    return df


def load_scada_from_zip(zip_bytes: bytes, zip_name: str):
    valid_frames = []
    ingest_records = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        for member in names:
            if member.endswith("/"):
                continue

            if not member.lower().endswith(".csv"):
                ingest_records.append(
                    {
                        "FileName": member,
                        "Status": "Skipped",
                        "RowsLoaded": 0,
                        "Message": "Not a CSV file",
                    }
                )
                continue

            try:
                file_bytes = zf.read(member)
                df = load_single_scada_bytes(file_bytes, member)
                valid_frames.append(df)
                ingest_records.append(
                    {
                        "FileName": member,
                        "Status": "Loaded",
                        "RowsLoaded": len(df),
                        "Message": "Loaded successfully",
                    }
                )
            except Exception as exc:
                ingest_records.append(
                    {
                        "FileName": member,
                        "Status": "Invalid",
                        "RowsLoaded": 0,
                        "Message": str(exc),
                    }
                )

    ingest_summary = pd.DataFrame(ingest_records)

    if not valid_frames:
        raise ValueError(f"No valid SCADA CSV files were found in archive '{zip_name}'.")

    merged = pd.concat(valid_frames, ignore_index=True)
    return merged, ingest_summary


@st.cache_data(show_spinner=False)
def load_scada_input(single_file_bytes, single_file_name, zip_file_bytes, zip_file_name):
    if zip_file_bytes is not None and zip_file_name is not None:
        scada_df, ingest_summary = load_scada_from_zip(zip_file_bytes, zip_file_name)
        upload_summary = {
            "upload_mode": "zip",
            "uploaded_file_count": int(len(ingest_summary)),
            "valid_file_count": int((ingest_summary["Status"] == "Loaded").sum()),
            "invalid_file_count": int((ingest_summary["Status"] == "Invalid").sum()),
            "skipped_file_count": int((ingest_summary["Status"] == "Skipped").sum()),
            "total_rows_uploaded": int(len(scada_df)),
            "archive_name": zip_file_name,
        }
        return scada_df, ingest_summary, upload_summary

    if single_file_bytes is not None and single_file_name is not None:
        scada_df = load_single_scada_bytes(single_file_bytes, single_file_name)
        ingest_summary = pd.DataFrame(
            [
                {
                    "FileName": single_file_name,
                    "Status": "Loaded",
                    "RowsLoaded": len(scada_df),
                    "Message": "Loaded successfully",
                }
            ]
        )
        upload_summary = {
            "upload_mode": "single_csv",
            "uploaded_file_count": 1,
            "valid_file_count": 1,
            "invalid_file_count": 0,
            "skipped_file_count": 0,
            "total_rows_uploaded": int(len(scada_df)),
            "archive_name": "",
        }
        return scada_df, ingest_summary, upload_summary

    raise ValueError("Please upload either a SCADA CSV file or a ZIP containing SCADA CSV files.")


@st.cache_data(show_spinner=False)
def load_reference_curves(file_bytes: bytes, file_name: str) -> dict[str, pd.DataFrame]:
    if not file_name.lower().endswith(".xlsx"):
        raise ValueError("Reference input must be an XLSX file.")

    try:
        raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name="Power Curve", header=None)
    except ValueError:
        raise ValueError("The Excel file must contain a sheet named 'Power Curve'.")

    if raw.shape[0] < 4:
        raise ValueError("Reference workbook Power Curve sheet does not contain enough rows.")

    site_row = raw.iloc[2]
    wind_speed = pd.to_numeric(raw.iloc[3:, 0], errors="coerce")
    curves = {}

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


def build_turbine_reason_tables(raw_scada: pd.DataFrame, date_filtered_scada: pd.DataFrame, filtered_final: pd.DataFrame):
    raw_counts = (
        raw_scada.groupby(WTG_COL)
        .size()
        .reset_index(name="UploadedRows")
        .rename(columns={WTG_COL: "WTG"})
    )

    date_counts = (
        date_filtered_scada.groupby(WTG_COL)
        .size()
        .reset_index(name="RowsInDateRange")
        .rename(columns={WTG_COL: "WTG"})
    )

    final_counts = (
        filtered_final.groupby("WTG")
        .size()
        .reset_index(name="ValidSamples")
    )

    merged = raw_counts.merge(date_counts, on="WTG", how="left").merge(final_counts, on="WTG", how="left")
    merged["RowsInDateRange"] = merged["RowsInDateRange"].fillna(0).astype(int)
    merged["ValidSamples"] = merged["ValidSamples"].fillna(0).astype(int)

    def classify_reason(row):
        if row["RowsInDateRange"] == 0:
            return "No rows available in selected date range"
        if row["ValidSamples"] == 0:
            return "Rows exist in date range but none passed filter/reference checks"
        if row["ValidSamples"] < LOW_DATA_THRESHOLD:
            return "Low number of valid rows after filtering"
        return "Working"

    merged["Reason"] = merged.apply(classify_reason, axis=1)

    missing_turbines = merged[merged["ValidSamples"] == 0][["WTG", "UploadedRows", "RowsInDateRange", "Reason"]].copy()
    low_data_turbines = merged[
        (merged["ValidSamples"] > 0) & (merged["ValidSamples"] < LOW_DATA_THRESHOLD)
    ][["WTG", "ValidSamples", "Reason"]].copy()
    low_data_turbines["Threshold"] = LOW_DATA_THRESHOLD
    low_data_turbines = low_data_turbines[["WTG", "ValidSamples", "Threshold", "Reason"]]

    working_turbines = merged[merged["ValidSamples"] > 0].copy()

    coverage_summary = {
        "uploaded_turbines": int(merged["WTG"].nunique()),
        "working_turbines": int((merged["ValidSamples"] > 0).sum()),
        "missing_turbines": int((merged["ValidSamples"] == 0).sum()),
        "low_data_turbines": int(((merged["ValidSamples"] > 0) & (merged["ValidSamples"] < LOW_DATA_THRESHOLD)).sum()),
    }

    return missing_turbines, low_data_turbines, working_turbines, coverage_summary


def run_analysis(scada: pd.DataFrame, ref: pd.DataFrame, raw_scada_full: pd.DataFrame | None = None) -> dict[str, pd.DataFrame | int]:
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
            ["After PCurveStsAve = 100 and WindSpeedStd < 1", int((mask_status & mask_wind_std).sum())],
            ["After all three required filters", int(mask.sum())],
        ],
        columns=["Filter Stage", "Record Count"],
    )

    rename_map = {
        WTG_COL: "WTG",
        TIME_COL: "DateTime",
        WIND_SPEED_COL: "WindSpeed_mps",
        POWER_COL: "ActualPower_kW",
    }
    if NACELLE_COL in filtered.columns:
        rename_map[NACELLE_COL] = "NacellePositionAve"
    if NACELLE_STD_COL in filtered.columns:
        rename_map[NACELLE_STD_COL] = "NacellePositionStd"

    filtered = filtered.rename(columns=rename_map)

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

    filtered["Performance_pct"] = np.where(
        filtered["ReferencePower_kW"] > 0,
        100 * filtered["ActualPower_kW"] / filtered["ReferencePower_kW"],
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
            MeanPerformance_pct=("Performance_pct", "mean"),
            PowerStdDev_kW=("ActualPower_kW", "std"),
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
            MeanPerformance_pct=("Performance_pct", "mean"),
            FirstTimestamp=("DateTime", "min"),
            LastTimestamp=("DateTime", "max"),
        )
        .reset_index()
    )

    wtg_summary["AggregateDeviation_pct"] = np.where(
        wtg_summary["MeanReferencePower_kW"] > 0,
        100 * (wtg_summary["MeanActualPower_kW"] - wtg_summary["MeanReferencePower_kW"])
        / wtg_summary["MeanReferencePower_kW"],
        np.nan,
    )

    status_comment = wtg_summary["MeanPerformance_pct"].apply(get_status_and_comment)
    wtg_summary["Status"] = status_comment.apply(lambda x: x[0])
    wtg_summary["PerformanceComment"] = status_comment.apply(lambda x: x[1])
    wtg_summary["Performance_pct"] = wtg_summary["MeanPerformance_pct"].round(2)

    wtg_summary = wtg_summary[
        [
            "WTG",
            "ValidSamples",
            "Performance_pct",
            "Status",
            "PerformanceComment",
            "MeanWindSpeed_mps",
            "MeanActualPower_kW",
            "MeanReferencePower_kW",
            "MeanDeviation_kW",
            "MeanSampleDeviation_pct",
            "AggregateDeviation_pct",
            "FirstTimestamp",
            "LastTimestamp",
        ]
    ].sort_values("Performance_pct")

    raw_for_reason = raw_scada_full.copy() if raw_scada_full is not None else scada.copy()
    missing_turbines, low_data_turbines, working_turbines, coverage_summary = build_turbine_reason_tables(
        raw_for_reason,
        scada,
        filtered,
    )

    results = {
        "filtered": filtered,
        "filter_counts": filter_counts,
        "wtg_bins": wtg_bins,
        "wtg_summary": wtg_summary,
        "ref": ref_working,
        "rows_passing": int(mask.sum()),
        "rows_compared": len(filtered),
        "wtg_count": int(filtered["WTG"].nunique()),
        "has_nacelle": "NacellePositionAve" in filtered.columns,
        "missing_turbines": missing_turbines,
        "low_data_turbines": low_data_turbines,
        "working_turbines_df": working_turbines,
        "turbine_coverage_summary": coverage_summary,
        "raw_wtg_count": int(raw_for_reason[WTG_COL].nunique()),
        "working_wtg_count": coverage_summary["working_turbines"],
        "missing_wtg_count": coverage_summary["missing_turbines"],
        "low_data_wtg_count": coverage_summary["low_data_turbines"],
    }

    if "NacellePositionAve" in filtered.columns:
        nacelle_filtered = filtered.dropna(subset=["NacellePositionAve"]).copy()
        nacelle_filtered["NacelleBand"] = nacelle_filtered["NacellePositionAve"].apply(get_nacelle_band)

        nacelle_bins = (
            nacelle_filtered.groupby(["WTG", "WindSpeedBin_mps"])
            .agg(
                Samples=("ActualPower_kW", "size"),
                MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
                MeanActualPower_kW=("ActualPower_kW", "mean"),
            )
            .reset_index()
        )

        nacelle_summary = (
            nacelle_filtered.groupby("WTG")
            .agg(
                ValidSamples=("ActualPower_kW", "size"),
                MeanWindSpeed_mps=("WindSpeed_mps", "mean"),
                MeanNacellePosition=("NacellePositionAve", "mean"),
                FirstTimestamp=("DateTime", "min"),
                LastTimestamp=("DateTime", "max"),
            )
            .reset_index()
        )
        nacelle_summary["NacelleBand"] = nacelle_summary["MeanNacellePosition"].apply(get_nacelle_band)
        nacelle_summary = nacelle_summary[
            [
                "WTG",
                "ValidSamples",
                "MeanWindSpeed_mps",
                "MeanNacellePosition",
                "NacelleBand",
                "FirstTimestamp",
                "LastTimestamp",
            ]
        ].sort_values("MeanNacellePosition")

        results["nacelle_bins"] = nacelle_bins
        results["nacelle_summary"] = nacelle_summary
        results["has_nacelle_std"] = "NacellePositionStd" in filtered.columns
    else:
        results["nacelle_bins"] = pd.DataFrame()
        results["nacelle_summary"] = pd.DataFrame()
        results["has_nacelle_std"] = False

    return results


def make_wtg_power_curve_plotly(ref, filtered_wtg, wtg_bins, wtg_name):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_wtg["WindSpeed_mps"],
            y=filtered_wtg["ActualPower_kW"],
            mode="markers",
            name="Power Samples",
            marker=dict(
                size=4,
                opacity=0.45,
                color="rgba(45, 115, 200, 0.55)",
                line=dict(color="rgba(25, 85, 160, 0.65)", width=0.5),
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ref["ReferenceWindSpeed_mps"],
            y=ref["ReferencePower_kW"],
            mode="lines",
            name="Reference Curve",
            line=dict(width=2, color="red"),
        )
    )

    if not wtg_bins.empty:
        fig.add_trace(
            go.Scatter(
                x=wtg_bins["MeanWindSpeed_mps"],
                y=wtg_bins["MeanActualPower_kW"],
                mode="lines",
                name=f"{wtg_name} mean by bin",
                line=dict(width=2, color="green"),
            )
        )

    fig.update_layout(
        title=f"{wtg_name} Power Curve",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def make_nacelle_plotly(ref, filtered_wtg, nacelle_bins, wtg_name):
    fig = go.Figure()

    if "NacellePositionStd" in filtered_wtg.columns:
        plot_df = filtered_wtg.dropna(
            subset=["WindSpeed_mps", "ActualPower_kW", "NacellePositionStd"]
        ).copy()
    else:
        plot_df = pd.DataFrame()

    if not plot_df.empty:
        custom_columns = ["NacellePositionStd"]
        if "NacellePositionAve" in plot_df.columns:
            custom_columns.append("NacellePositionAve")

        fig.add_trace(
            go.Scatter(
                x=plot_df["WindSpeed_mps"],
                y=plot_df["ActualPower_kW"],
                mode="markers",
                name="Samples (colour = Nacelle Position Std)",
                marker=dict(
                    size=4,
                    opacity=0.75,
                    color=plot_df["NacellePositionStd"],
                    colorscale=[
                        [0.0, "#2E7D32"],
                        [0.5, "#FDD835"],
                        [1.0, "#E53935"],
                    ],
                    showscale=True,
                    colorbar=dict(title="Nacelle<br>Position Std"),
                    line=dict(width=0),
                ),
                customdata=plot_df[custom_columns],
                hovertemplate=(
                    "Wind Speed: %{x:.2f} m/s"
                    "<br>Active Power: %{y:.2f} kW"
                    "<br>Nacelle Position Std: %{marker.color:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=ref["ReferenceWindSpeed_mps"],
            y=ref["ReferencePower_kW"],
            mode="lines",
            name="Reference Curve",
            line=dict(width=2, color="red"),
        )
    )

    if not nacelle_bins.empty:
        fig.add_trace(
            go.Scatter(
                x=nacelle_bins["MeanWindSpeed_mps"],
                y=nacelle_bins["MeanActualPower_kW"],
                mode="lines+markers",
                name=f"{wtg_name} mean by bin",
                line=dict(width=2, color="green"),
                marker=dict(size=5, color="green"),
            )
        )

    fig.update_layout(
        title=f"{wtg_name} Nacelle Std Coloured Power Curve",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def make_multi_turbine_comparison_plotly(ref, filtered, wtg_bins, selected_wtgs, show_scatter=False):
    fig = go.Figure()
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
        "#e377c2", "#17becf", "#bcbd22", "#d62728", "#7f7f7f",
        "#003f5c", "#ffa600", "#665191", "#a05195", "#d45087",
    ]

    if show_scatter:
        for i, wtg in enumerate(selected_wtgs):
            turbine_df = filtered[filtered["WTG"] == wtg]
            if turbine_df.empty:
                continue
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=turbine_df["WindSpeed_mps"],
                    y=turbine_df["ActualPower_kW"],
                    mode="markers",
                    name=f"{wtg} samples",
                    legendgroup=str(wtg),
                    marker=dict(size=4, opacity=0.5, color=color),
                    hovertemplate=(
                        f"Turbine: {wtg}<br>Wind Speed: %{{x:.2f}} m/s"
                        "<br>Active Power: %{y:.2f} kW<extra></extra>"
                    ),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=ref["ReferenceWindSpeed_mps"],
            y=ref["ReferencePower_kW"],
            mode="lines",
            name="Reference Curve",
            line=dict(width=2.5, color="red"),
            legendgroup="reference",
        )
    )

    for i, wtg in enumerate(selected_wtgs):
        turbine_bins = wtg_bins[wtg_bins["WTG"] == wtg]
        if turbine_bins.empty:
            continue
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=turbine_bins["MeanWindSpeed_mps"],
                y=turbine_bins["MeanActualPower_kW"],
                mode="lines+markers",
                name=f"{wtg} mean by bin",
                legendgroup=str(wtg),
                line=dict(width=2.5, color=color),
                marker=dict(size=5, color=color),
                hovertemplate=(
                    f"Turbine: {wtg}<br>Mean Wind Speed: %{{x:.2f}} m/s"
                    "<br>Mean Active Power: %{y:.2f} kW<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Multi-Turbine Power Curve Comparison",
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        height=650,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=45, r=45, t=90, b=45),
        legend=dict(orientation="h", y=-0.2, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


st.sidebar.header("Upload Files")
uploaded_scada = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])
uploaded_scada_zip = st.sidebar.file_uploader("Upload SCADA ZIP", type=["zip"])
uploaded_ref = st.sidebar.file_uploader("Upload Reference Excel", type=["xlsx"])

if uploaded_scada is None and uploaded_scada_zip is None:
    st.info("Please upload either the SCADA CSV file or a ZIP containing SCADA CSV files.")
    st.stop()

if uploaded_ref is None:
    st.info("Please upload the reference Excel file.")
    st.stop()

try:
    scada, file_ingest_summary, upload_summary = load_scada_input(
        uploaded_scada.getvalue() if uploaded_scada else None,
        uploaded_scada.name if uploaded_scada else None,
        uploaded_scada_zip.getvalue() if uploaded_scada_zip else None,
        uploaded_scada_zip.name if uploaded_scada_zip else None,
    )
    reference_curves = load_reference_curves(uploaded_ref.getvalue(), uploaded_ref.name)
except Exception as exc:
    st.exception(exc)
    st.stop()

curve_names = sorted(reference_curves.keys())
selected_curve = st.sidebar.selectbox("Select reference curve", curve_names)

scada_filtered = scada.copy()
valid_time = scada[TIME_COL].dropna()
date_text = "All available dates"

if not valid_time.empty:
    data_min_date = valid_time.min().date()
    data_max_date = valid_time.max().date()
    today_value = data_max_date

    date_filter_type = st.sidebar.radio(
        "Date Filter",
        ["Today", "Last 7 Days", "This Month", "Last Month", "Manual Calendar"],
        index=4,
    )

    if date_filter_type == "Today":
        start_date = today_value
        end_date = today_value
    elif date_filter_type == "Last 7 Days":
        start_date = today_value - timedelta(days=6)
        end_date = today_value
    elif date_filter_type == "This Month":
        start_date = today_value.replace(day=1)
        end_date = today_value
    elif date_filter_type == "Last Month":
        start_date, end_date = get_last_month_range(today_value)
    else:
        manual_range = st.sidebar.date_input(
            "Date range",
            value=(data_min_date, data_max_date),
            min_value=data_min_date,
            max_value=data_max_date,
        )

        if isinstance(manual_range, (tuple, list)) and len(manual_range) == 2:
            start_date, end_date = manual_range
        else:
            start_date, end_date = data_min_date, data_max_date

    start_date = max(start_date, data_min_date)
    end_date = min(end_date, data_max_date)

    scada_filtered = scada_filtered[
        scada_filtered[TIME_COL].dt.date.between(start_date, end_date)
    ].copy()

    date_text = f"{start_date} to {end_date}"

site_name = selected_curve

header_left, header_right = st.columns([1, 4])

with header_left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)

with header_right:
    st.title("Power Curve Analytic Report")
    st.markdown(f"**Site Name:** {site_name}")
    st.markdown(f"**Date Range:** {date_text}")

if scada_filtered.empty:
    st.warning("No SCADA data available after date filtering.")
    st.stop()

try:
    results = run_analysis(scada_filtered, reference_curves[selected_curve], raw_scada_full=scada)
    results["file_ingest_summary"] = file_ingest_summary
    results["upload_summary"] = upload_summary
except Exception as exc:
    st.exception(exc)
    st.stop()

st.subheader("Upload Summary")
u1, u2, u3, u4 = st.columns(4)
u1.metric("Uploaded files", upload_summary["uploaded_file_count"])
u2.metric("Valid SCADA files", upload_summary["valid_file_count"])
u3.metric("Invalid files", upload_summary["invalid_file_count"])
u4.metric("Rows uploaded", f"{upload_summary['total_rows_uploaded']:,}")

if upload_summary["upload_mode"] == "zip":
    st.caption(f"Archive: {upload_summary['archive_name']}")

st.subheader("Turbine Coverage Summary")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Uploaded turbines", results["turbine_coverage_summary"]["uploaded_turbines"])
t2.metric("Working turbines", results["turbine_coverage_summary"]["working_turbines"])
t3.metric("Missing turbines", results["turbine_coverage_summary"]["missing_turbines"])
t4.metric("Low-data turbines", results["turbine_coverage_summary"]["low_data_turbines"])

c1, c2, c3 = st.columns(3)
c1.metric("Rows passing filters", f"{results['rows_passing']:,}")
c2.metric("Rows compared", f"{results['rows_compared']:,}")
c3.metric("WTGs analysed", f"{results['wtg_count']:,}")

st.subheader("File Ingest Details")
st.dataframe(file_ingest_summary, use_container_width=True, hide_index=True)

st.subheader("Data Availability")
da1, da2 = st.columns(2)

with da1:
    st.markdown("### Missing Turbines")
    if results["missing_turbines"].empty:
        st.success("No missing turbines.")
    else:
        st.dataframe(results["missing_turbines"], use_container_width=True, hide_index=True)

with da2:
    st.markdown("### Low-Data Turbines")
    if results["low_data_turbines"].empty:
        st.success("No low-data turbines.")
    else:
        st.dataframe(results["low_data_turbines"], use_container_width=True, hide_index=True)

st.subheader("PDF Downloads")
pdf_col1, pdf_col2 = st.columns(2)

with pdf_col1:
    if KALEIDO_AVAILABLE and REPORTLAB_AVAILABLE:
        if st.button("Prepare Full Dashboard PDF"):
            with st.spinner("Generating dashboard PDF..."):
                try:
                    st.session_state["dashboard_pdf_bytes"] = build_dashboard_pdf(site_name, date_text, results)
                    st.success("Dashboard PDF is ready.")
                except Exception as exc:
                    st.error(f"Dashboard PDF could not be generated: {exc}")

        if "dashboard_pdf_bytes" in st.session_state:
            st.download_button(
                "Download Full Dashboard PDF",
                data=st.session_state["dashboard_pdf_bytes"],
                file_name="power_curve_dashboard_report.pdf",
                mime="application/pdf",
            )
    else:
        st.info(
            f"Dashboard PDF export requires kaleido and reportlab. "
            f"Kaleido: {KALEIDO_IMPORT_ERROR or 'OK'} | Reportlab: {REPORTLAB_IMPORT_ERROR or 'OK'}"
        )

with pdf_col2:
    if results["has_nacelle"]:
        st.info("Nacelle section is included automatically inside the full dashboard PDF when data exists.")
    else:
        st.info("Nacelle section is not included because nacelle data is unavailable.")

if results["has_nacelle"]:
    tab1, tab2, tab3 = st.tabs([
        "Power Curve",
        "Nacelle Position vs Wind Speed",
        "Multi-Turbine Comparison",
    ])
else:
    tab1, tab3 = st.tabs(["Power Curve", "Multi-Turbine Comparison"])
    tab2 = None
    st.warning(
        f"Nacelle tab is hidden because column '{NACELLE_COL}' was not found in the uploaded SCADA file."
    )

with tab1:
    st.subheader("Filter Summary")
    st.dataframe(results["filter_counts"], use_container_width=True, hide_index=True)

    st.subheader("WTG Summary")
    st.table(style_wtg_summary(results["wtg_summary"]))

    all_wtgs = sorted(results["filtered"]["WTG"].dropna().unique().tolist())
    summary_lookup = results["wtg_summary"].set_index("WTG")

    st.subheader(f"All Turbine Power Curves ({len(all_wtgs)})")

    for wtg in all_wtgs:
        filtered_wtg = results["filtered"][results["filtered"]["WTG"] == wtg].copy()
        wtg_bins = results["wtg_bins"][results["wtg_bins"]["WTG"] == wtg].copy()

        if filtered_wtg.empty:
            continue

        perf = summary_lookup.loc[wtg, "Performance_pct"] if wtg in summary_lookup.index else np.nan
        dev = summary_lookup.loc[wtg, "AggregateDeviation_pct"] if wtg in summary_lookup.index else np.nan
        comment = summary_lookup.loc[wtg, "PerformanceComment"] if wtg in summary_lookup.index else "Performance comment not available"

        render_status_badge(wtg, perf, dev, comment)

        fig = make_wtg_power_curve_plotly(results["ref"], filtered_wtg, wtg_bins, wtg)
        st.plotly_chart(fig, use_container_width=True)

if tab2 is not None:
    with tab2:
        if results["nacelle_summary"].empty:
            st.warning("Nacelle position data is not available after filtering.")
        else:
            st.subheader("Nacelle Position Summary")
            st.table(style_nacelle_summary(results["nacelle_summary"]))

            all_wtgs = sorted(results["nacelle_summary"]["WTG"].dropna().unique().tolist())
            nacelle_lookup = results["nacelle_summary"].set_index("WTG")

            st.subheader(f"All Turbine Nacelle Position Graphs ({len(all_wtgs)})")

            for wtg in all_wtgs:
                filtered_wtg = results["filtered"][results["filtered"]["WTG"] == wtg].copy()
                filtered_wtg = filtered_wtg.dropna(subset=["NacellePositionAve"]).copy()
                filtered_wtg["NacelleBand"] = filtered_wtg["NacellePositionAve"].apply(get_nacelle_band)

                nacelle_bins = results["nacelle_bins"][results["nacelle_bins"]["WTG"] == wtg].copy()

                if filtered_wtg.empty:
                    continue

                avg_nacelle = nacelle_lookup.loc[wtg, "MeanNacellePosition"] if wtg in nacelle_lookup.index else np.nan
                nacelle_band = nacelle_lookup.loc[wtg, "NacelleBand"] if wtg in nacelle_lookup.index else "NA"

                render_nacelle_badge(wtg, avg_nacelle, nacelle_band)

                fig = make_nacelle_plotly(results["ref"], filtered_wtg, nacelle_bins, wtg)
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Compare Multiple Turbines")
    comparison_wtgs = sorted(results["filtered"]["WTG"].dropna().unique().tolist())
    if not comparison_wtgs:
        st.warning("No turbines are available for comparison after filtering.")
    else:
        default_selection = comparison_wtgs[:min(3, len(comparison_wtgs))]
        selected_wtgs = st.multiselect(
            "Select turbine names to compare",
            options=comparison_wtgs,
            default=default_selection,
            help="Select turbines to compare their power curves on one graph.",
            key="multi_turbine_comparison_selection",
        )
        if not selected_wtgs:
            st.info("Select at least one turbine to display the comparison graph.")
        else:
            show_comparison_scatter = st.checkbox(
                "Show scatter points",
                value=False,
                help="Turn this on to display individual SCADA samples for each selected turbine.",
                key="multi_turbine_show_scatter",
            )
            comparison_filtered = results["filtered"][
                results["filtered"]["WTG"].isin(selected_wtgs)
            ].copy()
            comparison_bins = results["wtg_bins"][
                results["wtg_bins"]["WTG"].isin(selected_wtgs)
            ].copy()
            comparison_fig = make_multi_turbine_comparison_plotly(
                results["ref"], comparison_filtered, comparison_bins, selected_wtgs,
                show_scatter=show_comparison_scatter,
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            st.caption(
                "Each turbine has its own colour and name in the legend. The red line is the "
                "reference curve. Enable 'Show scatter points' whenever you want to see individual samples."
            )
