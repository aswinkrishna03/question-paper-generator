from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import time
import os

def generate_pdf(paper):
    output_dir = "generated_papers"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"Question_Paper_{int(time.time())}.pdf"
    filepath = os.path.join(output_dir, filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    left_margin = 50
    right_margin = 50
    max_width = width - left_margin - right_margin
    y = height - 50

    for line in paper:
        wrapped_lines = simpleSplit(line, "Helvetica", 12, max_width)

        for wrap_line in wrapped_lines:
            if y < 50:
                c.showPage()
                y = height - 50

            c.drawString(left_margin, y, wrap_line)
            y -= 18

        y -= 5  # spacing between questions

    c.save()
    return filepath