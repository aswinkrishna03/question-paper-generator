from flask import Flask, render_template, request
from pypdf import PdfReader
import os
import re

from classifier import classify_sentence
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

        # 1. Clear old questions
        clear_questions()

        # 2. Save uploaded PDF
        pdf_file = request.files["pdf"]
        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
        pdf_file.save(file_path)

        # 3. Extract text from PDF
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        print("PDF TEXT SAMPLE:")
        print(text[:500])

        # 4. Clean text (light cleaning only)
        text = text.replace("\n", " ")
        text = re.sub(r"Module\s*-?\s*\d+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"Module\d+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[■•–]", " ", text)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 5. Split into topic lines
        lines = re.split(r"[.:]", text)

        count = 0

        for line in lines:
            line = line.strip()
            if len(line) < 6:
                continue

            # 6. Split compound topics
            topics = re.split(r"-| and |,", line)

            for topic in topics:
                topic = topic.strip()
                if len(topic) < 6:
                    continue

                topic_lower = topic.lower()

                # Normalize common concepts
                if "dma" in topic_lower:
                    topic = "DMA transfer"
                elif "control bus" in topic_lower:
                    topic = "Control bus"

                # 7. Generate question
                q_type = classify_sentence(topic)
                question = generate_question(topic, q_type)

                if q_type == "definition":
                    marks = 2
                elif q_type == "comparison":
                    marks = 5
                elif q_type == "analysis":
                    marks = 10
                else:
                    marks = 5

                insert_question(question, q_type, marks)
                count += 1

                if count >= 10:
                    break

            if count >= 10:
                break

        # 8. Generate question paper PDF
        paper = generate_question_paper()
        generate_pdf(paper)

        return "Question Paper Generated Successfully. Check Question_Paper.pdf"

    return render_template("upload.html")


print("APP FILE LOADED")

if __name__ == "__main__":
    print("STARTING FLASK SERVER")
    app.run(debug=True)
