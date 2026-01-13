from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_pdf(questions):
    pdf = canvas.Canvas("Question_Paper.pdf", pagesize=A4)
    width, height = A4

    text = pdf.beginText(50, height - 50)
    text.setFont("Helvetica-Bold", 14)
    text.textLine("AUTOMATIC QUESTION PAPER")
    text.textLine("")

    text.setFont("Helvetica", 12)
    for i, q in enumerate(questions, 1):
        text.textLine(f"{i}. {q}")
        text.textLine("")

    pdf.drawText(text)
    pdf.save()
