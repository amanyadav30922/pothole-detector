import streamlit as st
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
from PIL import Image
import json, os
import pandas as pd
import numpy as np

st.set_page_config(page_title="Pothole Detector", layout="wide")
st.title("🕳️ Pothole Detection Dashboard")
st.markdown("AI-powered pothole detection and mapping system")

# Load detections
with open('detections.json') as f:
    detections = json.load(f)

# Sidebar stats
st.sidebar.title("📊 Statistics")
st.sidebar.metric("Total Locations",    len(detections))
st.sidebar.metric("🔴 High Severity",   sum(1 for d in detections if d['severity'] == 'High'))
st.sidebar.metric("🟡 Medium Severity", sum(1 for d in detections if d['severity'] == 'Medium'))
st.sidebar.metric("🟢 Low Severity",    sum(1 for d in detections if d['severity'] == 'Low'))

# Tabs
tab1, tab2, tab3 = st.tabs(["🗺️ Map", "📸 Live Detection", "📋 Data Table"])

# ─── Tab 1: Map ───────────────────────────────────────────────
with tab1:
    st.subheader("Pothole Hotspot Map")

    severity_filter = st.selectbox("Filter by severity", ["All", "High", "Medium", "Low"])

    filtered = detections if severity_filter == "All" else [
        d for d in detections if d['severity'] == severity_filter
    ]

    m = folium.Map(location=[28.6, 77.2], zoom_start=9)
    colors = {"High": "red", "Medium": "orange", "Low": "green"}

    for d in filtered:
        folium.CircleMarker(
            location=[d['lat'], d['lon']],
            radius=8,
            color=colors[d['severity']],
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>City:</b> {d['city']}<br>"
                f"<b>Potholes:</b> {d['potholes']}<br>"
                f"<b>Severity:</b> {d['severity']}<br>"
                f"<b>Time:</b> {d['timestamp']}",
                max_width=200
            )
        ).add_to(m)

    st_folium(m, width=900, height=500)
    st.caption("🔴 High  🟡 Medium  🟢 Low — Click any dot for details")

# ─── Tab 2: Live Detection ────────────────────────────────────
with tab2:
    st.subheader("📸 Upload Road Image — GPS Auto Detected")

    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error("❌ best.pt not found — place it in the same folder as app.py")
    else:
        model = YOLO(model_path)

        uploaded = st.file_uploader("Upload a road image", type=['jpg', 'jpeg', 'png'])

        if uploaded:
            img = Image.open(uploaded)

            # ── Auto read GPS from image EXIF ──
            def get_gps(image_file):
                try:
                    import piexif
                    image = Image.open(image_file)
                    exif_data = piexif.load(image.info.get('exif', b''))
                    gps = exif_data.get('GPS', {})
                    if not gps:
                        return None
                    def to_decimal(vals):
                        d = vals[0][0] / vals[0][1]
                        m = vals[1][0] / vals[1][1]
                        s = vals[2][0] / vals[2][1]
                        return d + m/60 + s/3600
                    lat = to_decimal(gps[2])
                    lon = to_decimal(gps[4])
                    if gps[1] == b'S': lat = -lat
                    if gps[3] == b'W': lon = -lon
                    return lat, lon
                except:
                    return None

            coords = get_gps(uploaded)

            # ── Run YOLOv8 detection ──
            results     = model(img, conf=0.3)
            res_plotted = results[0].plot()
            num_potholes = len(results[0].boxes)
            boxes        = results[0].boxes

            # ── Show original + detected images ──
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            with col2:
                st.image(
                    res_plotted[:, :, ::-1],
                    caption=f"Detected: {num_potholes} pothole(s)",
                    use_column_width=True
                )

            if num_potholes > 0:
                st.error(f"⚠️ {num_potholes} pothole(s) detected!")
            else:
                st.success("✅ No potholes detected — Road looks good!")

            st.divider()

            # ── GPS Section ──
            st.subheader("📍 Location")

            if coords:
                lat, lon = coords
                st.success("✅ GPS auto-detected from photo!")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Latitude",  f"{lat:.6f}")
                with col_b:
                    st.metric("Longitude", f"{lon:.6f}")

                # Build mini map with camera pin + pothole dots
                mini_map = folium.Map(location=[lat, lon], zoom_start=16)
                folium.Marker(
                    [lat, lon],
                    popup="Photo taken here",
                    icon=folium.Icon(color='blue', icon='camera')
                ).add_to(mini_map)

                colors_sev  = {"High": "red", "Medium": "orange", "Low": "green"}
                pothole_list = []

                for i, box in enumerate(boxes):
                    cx, cy = box.xywh[0][0].item(), box.xywh[0][1].item()
                    bw, bh = box.xywh[0][2].item(), box.xywh[0][3].item()
                    conf   = box.conf[0].item()
                    iw, ih = img.size

                    # Convert pixel offset to GPS offset
                    offset_x = (cx - iw / 2) / iw * 0.001
                    offset_y = (cy - ih / 2) / ih * 0.001
                    pot_lat  = lat - offset_y
                    pot_lon  = lon + offset_x

                    area = (bw * bh) / (iw * ih)
                    sev  = "High" if area > 0.05 else "Medium" if area > 0.02 else "Low"

                    pothole_list.append({
                        "id":         i + 1,
                        "lat":        pot_lat,
                        "lon":        pot_lon,
                        "severity":   sev,
                        "confidence": conf
                    })

                    folium.CircleMarker(
                        location=[pot_lat, pot_lon],
                        radius=12,
                        color=colors_sev[sev],
                        fill=True,
                        fill_opacity=0.8,
                        popup=f"Pothole #{i+1} | {sev} | {conf:.0%} confidence"
                    ).add_to(mini_map)

                st_folium(mini_map, width=700, height=350, key="mini_map")

                # ── Save section ──
                if num_potholes > 0:
                    st.divider()
                    city = st.text_input("📌 Area/City name:", value="Unknown")

                    col_s1, col_s2 = st.columns(2)

                    with col_s1:
                        if st.button("💾 Save ALL to Map"):
                            from datetime import datetime
                            with open('detections.json', 'r') as f:
                                all_det = json.load(f)

                            for p in pothole_list:
                                all_det.append({
                                    "image":      uploaded.name,
                                    "lat":        p["lat"],
                                    "lon":        p["lon"],
                                    "city":       city,
                                    "potholes":   1,
                                    "severity":   p["severity"],
                                    "confidence": round(p["confidence"], 2),
                                    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })

                            with open('detections.json', 'w') as f:
                                json.dump(all_det, f, indent=2)

                            st.success(f"✅ {len(pothole_list)} pothole(s) saved to map!")
                            st.balloons()
                            st.info("Go to 🗺️ Map tab to see them!")

                    with col_s2:
                        df_new = pd.DataFrame(pothole_list)
                        csv    = df_new.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download this detection",
                            csv,
                            f"detection_{uploaded.name}.csv",
                            "text/csv"
                        )

            else:
                # No GPS found in image
                st.warning("⚠️ No GPS found in this image")
                st.info("""
                **To get GPS in your photos, choose one of these:**
                - Use our `streetview_gps.py` script to embed GPS into any photo
                - Take photos with your phone with Location turned ON
                - Transfer photos via USB cable (NOT WhatsApp — it removes GPS!)
                """)

# ─── Tab 3: Data Table + PDF ─────────────────────────────────
with tab3:
    st.subheader("All Detections")

    df = pd.DataFrame(detections)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "pothole_detections.csv", "text/csv")

    st.divider()
    st.subheader("📄 Generate PDF Report")

    if st.button("Generate PDF Report"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.units import inch
            from datetime import datetime
            import io

            buffer = io.BytesIO()
            doc    = SimpleDocTemplate(
                buffer, pagesize=A4,
                rightMargin=inch*0.75, leftMargin=inch*0.75,
                topMargin=inch*0.75,   bottomMargin=inch*0.75
            )

            styles   = getSampleStyleSheet()
            elements = []

            # Title
            title_style = ParagraphStyle(
                'title', parent=styles['Title'],
                fontSize=22, textColor=colors.HexColor('#1a1a2e'), spaceAfter=6
            )
            elements.append(Paragraph("Pothole Detection Report", title_style))

            sub_style = ParagraphStyle(
                'sub', parent=styles['Normal'],
                fontSize=11, textColor=colors.grey, spaceAfter=20
            )
            elements.append(Paragraph(
                f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} | "
                f"AI-Powered Road Safety System",
                sub_style
            ))
            elements.append(Spacer(1, 0.2*inch))

            # Summary table
            total_potholes = sum(d['potholes'] for d in detections)
            high   = sum(1 for d in detections if d['severity'] == 'High')
            medium = sum(1 for d in detections if d['severity'] == 'Medium')
            low    = sum(1 for d in detections if d['severity'] == 'Low')

            summary_data = [
                ['Metric',                    'Value'],
                ['Total Locations Scanned',   str(len(detections))],
                ['Total Potholes Found',      str(total_potholes)],
                ['High Severity Locations',   str(high)],
                ['Medium Severity Locations', str(medium)],
                ['Low Severity Locations',    str(low)],
            ]

            summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND',     (0, 0), (-1,  0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR',      (0, 0), (-1,  0), colors.white),
                ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
                ('FONTSIZE',       (0, 0), (-1,  0), 12),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4ff')]),
                ('FONTNAME',       (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE',       (0, 1), (-1, -1), 11),
                ('GRID',           (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ALIGN',          (1, 0), ( 1, -1), 'CENTER'),
                ('PADDING',        (0, 0), (-1, -1), 8),
            ]))

            elements.append(Paragraph("Summary", styles['Heading2']))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.3*inch))

            # Top 10 worst locations
            elements.append(Paragraph("Top 10 Worst Locations", styles['Heading2']))
            elements.append(Spacer(1, 0.1*inch))

            top10       = sorted(detections, key=lambda x: x['potholes'], reverse=True)[:10]
            detail_data = [['#', 'City', 'Potholes', 'Severity', 'Coordinates']]
            for i, d in enumerate(top10, 1):
                detail_data.append([
                    str(i), d['city'], str(d['potholes']),
                    d['severity'], f"{d['lat']:.4f}, {d['lon']:.4f}"
                ])

            detail_table = Table(detail_data, colWidths=[0.4*inch, 1.5*inch, 1*inch, 1*inch, 2*inch])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND',     (0, 0), (-1,  0), colors.HexColor('#e63946')),
                ('TEXTCOLOR',      (0, 0), (-1,  0), colors.white),
                ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
                ('FONTSIZE',       (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff5f5')]),
                ('GRID',           (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ALIGN',          (0, 0), (-1, -1), 'CENTER'),
                ('PADDING',        (0, 0), (-1, -1), 7),
            ]))
            elements.append(detail_table)
            elements.append(Spacer(1, 0.3*inch))

            # All detections table
            elements.append(Paragraph("All Detected Locations", styles['Heading2']))
            elements.append(Spacer(1, 0.1*inch))

            all_data = [['City', 'Potholes', 'Severity', 'Lat', 'Lon', 'Timestamp']]
            for d in detections:
                all_data.append([
                    d['city'], str(d['potholes']), d['severity'],
                    f"{d['lat']:.4f}", f"{d['lon']:.4f}", d['timestamp']
                ])

            all_table = Table(
                all_data,
                colWidths=[1.2*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.8*inch]
            )
            all_table.setStyle(TableStyle([
                ('BACKGROUND',     (0, 0), (-1,  0), colors.HexColor('#457b9d')),
                ('TEXTCOLOR',      (0, 0), (-1,  0), colors.white),
                ('FONTNAME',       (0, 0), (-1,  0), 'Helvetica-Bold'),
                ('FONTSIZE',       (0, 0), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1faee')]),
                ('GRID',           (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
                ('ALIGN',          (0, 0), (-1, -1), 'CENTER'),
                ('PADDING',        (0, 0), (-1, -1), 5),
            ]))
            elements.append(all_table)
            elements.append(Spacer(1, 0.3*inch))

            # Footer
            footer_style = ParagraphStyle(
                'footer', parent=styles['Normal'],
                fontSize=9, textColor=colors.grey
            )
            elements.append(Paragraph(
                "This report was automatically generated by the AI Pothole Detection System. "
                "Please submit this report to your local municipal authority for road repair action.",
                footer_style
            ))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                label="⬇️ Download PDF Report",
                data=buffer,
                file_name=f"pothole_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
            st.success("✅ PDF generated! Click the button above to download.")

        except ImportError:
            st.error("❌ reportlab not installed. Run: pip install reportlab")