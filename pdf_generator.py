from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import time
import os

def generate_pdf(paper):
    output_dir = "generated_papers"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"Question_Paper_{int(time.time())}.pdf"
    filepath = os.path.join(output_dir, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4
    y = height - 50

    for line in paper:
        c.drawString(50, y, line)
        y -= 20

        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    return filepath
