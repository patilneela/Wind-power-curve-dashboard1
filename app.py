            res["turbine"],
            res["avg_nacelle"],
            res["nacelle_sample_count"],
        )
        st.plotly_chart(fig_nacelle, use_container_width=True)
        st.caption(f"Nacelle Sample Formula: X = {wind_col}, Y = {nacelle_col}")
        st.caption(f"Nacelle Samples Count: {res['nacelle_sample_count']}")

    st.markdown("### Analysis")
    st.code(res["comment"])
    st.divider()

    figures.append((t, fig_power, res["comment"]))

    results.append(
        {
            "Turbine": t,
            "Deviation_%": None if res["avg_dev"] is None else round(res["avg_dev"], 2),
            "Avg_Nacelle": None if res["avg_nacelle"] is None else round(res["avg_nacelle"], 2),
            "Nacelle_Band": get_nacelle_band(res["avg_nacelle"]),
            "Power_Samples": res["power_sample_count"],
            "Deviation_Filtered_Samples": res.get("deviation_filter_rows", 0),
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
                f"Power Samples: {row['Power_Samples']} | "
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
