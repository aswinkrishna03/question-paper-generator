from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import os
import time
from xml.sax.saxutils import escape


def generate_pdf(lines):

    os.makedirs("generated_papers", exist_ok=True)

    filename = f"Question_Paper_{int(time.time())}.pdf"
    filepath = os.path.join("generated_papers", filename)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filepath, pagesize=A4)

    elements = []

    for index, line in enumerate(lines):
        if isinstance(line, dict) and line.get("type") == "header_meta":
            meta_table = Table(
                [[
                    Paragraph(str(line.get("left", "")), styles["Normal"]),
                    Paragraph(str(line.get("right", "")), styles["Normal"]),
                ]],
                colWidths=[doc.width / 2, doc.width / 2],
            )
            meta_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (0, 0), "LEFT"),
                        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
                    ]
                )
            )
            elements.append(meta_table)
            elements.append(Spacer(1, 10))
            continue

        if index == 0:
            elements.append(Paragraph(f"<b>{escape(str(line))}</b>", styles["Title"]))
            elements.append(Spacer(1, 10))
            continue

        elements.append(Paragraph(escape(str(line)), styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)

    return filename
