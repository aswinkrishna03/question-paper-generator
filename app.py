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

        # 🔁 Clear old questions
        clear_questions()
        seen_topics = set()

        # 📥 Save uploaded PDF
        pdf_file = request.files["pdf"]
        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
        pdf_file.save(file_path)

        # 📄 Extract text from PDF
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        # 🧹 Clean extracted text
        text = text.replace("\n", " ")
        text = re.sub(r"[■•–]", " ", text)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # ✂ Split into sentences
        lines = re.split(r"[.:]", text)

        count = 0

        for line in lines:
            line = line.strip()
            if len(line) < 6:
                continue

            topics = re.split(r"-| and |,", line)

            for topic in topics:
                topic = topic.strip()
                topic_lower = topic.lower()

                # ==============================
                # 🚫 FILTER UNWANTED TEXT
                # ==============================

                if len(topic) < 6:
                    continue

                # Course codes (MCN201, etc.)
                if re.search(r"[A-Z]{3,}\d{2,}", topic):
                    continue

                # Full uppercase headings
                if topic.isupper():
                    continue

                # Instructional / prompt leakage
                instruction_phrases = [
                    "generate a question",
                    "from this text",
                    "its sources",
                    "its uses",
                    "its applications",
                    "explain the following"
                ]
                if any(p in topic_lower for p in instruction_phrases):
                    continue

                # Repeated page headers
                if "sun is the primary source" in topic_lower:
                    continue

                # Duplicate topic protection
                if topic_lower in seen_topics:
                    continue
                seen_topics.add(topic_lower)

                # Repeated-word noise
                words = topic.split()
                if len(words) > 3 and len(set(words)) < len(words) / 2:
                    continue

                # Meaningless generic words
                if topic_lower in ["resources", "engineering", "syllabus"]:
                    continue

                # ==============================
                # NORMALIZE COMMON TERMS
                # ==============================
                if "dma" in topic_lower:
                    topic = "DMA transfer"
                elif "control bus" in topic_lower:
                    topic = "Control bus"

                # ==============================
                # GENERATE QUESTION
                # ==============================
                q_type = classify_sentence(topic)
                question = generate_question(topic, q_type)

                marks = 2 if q_type == "definition" else 5
                insert_question(question, q_type, marks)

                count += 1
                if count >= 10:
                    break

            if count >= 10:
                break

        # 📑 Generate Question Paper & PDF
        paper = generate_question_paper()
        filepath = generate_pdf(paper)

        return f"Question Paper Generated Successfully: {filepath}"

    return render_template("upload.html")


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
