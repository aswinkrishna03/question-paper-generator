from flask import Flask, render_template, request
from pypdf import PdfReader
import os
import re

from question_generator import generate_question
from database import insert_question, clear_questions
from paper_generator import generate_question_paper
from pdf_generator import generate_pdf

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def upload_pdf():
    if request.method == "POST":

        clear_questions()

        pdf_file = request.files["pdf"]
        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
        pdf_file.save(file_path)

        reader = PdfReader(file_path)
        full_text = ""

        # Extract text
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + " "

        # Basic cleaning
        full_text = re.sub(r'http\S+|www\.\S+|\S+\.(com|in|org|net)', ' ', full_text)
        full_text = re.sub(r'\b[A-Z]{2,}\d{2,}\b', ' ', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        # 🔥 Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', full_text)

        stored_questions = set()
        count = 0

        # 🔥 Group every 3 sentences into one paragraph
        for i in range(0, len(sentences), 3):

            paragraph = " ".join(sentences[i:i+3]).strip()

            if len(paragraph.split()) < 25:
                continue

            question = generate_question(paragraph, "general")

            if not question:
                continue

            if question in stored_questions:
                continue

            insert_question(question, "general", 5)
            stored_questions.add(question)

            count += 1

            if count >= 5:
                break

        paper = generate_question_paper()
        filepath = generate_pdf(paper)

        return f"Question Paper Generated Successfully. Saved at: {filepath}"

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)