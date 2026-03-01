from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from pypdf import PdfReader
import os
import re
from datetime import datetime

from question_generator import (
    generate_question,
    generate_short_question,
    generate_long_question
)

from database import insert_question, clear_questions
from paper_generator import generate_question_paper
from pdf_generator import generate_pdf

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "generated_papers"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Simple in-memory users (for demo)
users = {}

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("upload_pdf"))
        else:
            return "Invalid Credentials"

    return render_template("login.html")


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users[username] = password
        return redirect(url_for("login"))

    return render_template("register.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


# ---------------- HOME + GENERATION ----------------
@app.route("/", methods=["GET", "POST"])
def upload_pdf():

    if not session.get("username"):
        return redirect(url_for("login"))

    if request.method == "POST":

        clear_questions()

        pdf_file = request.files["pdf"]
        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
        pdf_file.save(file_path)

        reader = PdfReader(file_path)
        full_text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + " "

        # ---------------- CLEAN TEXT ----------------
        full_text = re.sub(r'http\S+|www\.\S+|\S+\.(com|in|org|net)', ' ', full_text)
        full_text = re.sub(r'\b[A-Z]{2,}\d+\b', ' ', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()

        sentences = re.split(r'(?<=[.!?]) +', full_text)

        concept_count = 0

        for i in range(0, len(sentences), 3):

            paragraph = " ".join(sentences[i:i+3]).strip()

            if len(paragraph.split()) < 20:
                continue

            # Reject headings / garbage
            if any(word in paragraph.lower() for word in [
                "module", "unit", "chapter", "semester",
                "figure", "diagram", "table"
            ]):
                continue

            # Extract concept
            concept = generate_question(paragraph, "general")

            if not concept:
                continue

            # Generate one short & one long question per concept
            short_q = generate_short_question(concept)
            long_q = generate_long_question(concept)

            insert_question(short_q, "short", 3)
            insert_question(long_q, "long", 10)

            concept_count += 1

            if concept_count >= 5:
                break

        paper = generate_question_paper()
        filepath = generate_pdf(paper)

        file_name = os.path.basename(filepath)

        return render_template("result.html", file_name=file_name)

    return render_template("upload.html")


# ---------------- HISTORY ----------------
@app.route("/history")
def history():

    if not session.get("username"):
        return redirect(url_for("login"))

    files = os.listdir(GENERATED_FOLDER)

    papers = []
    for file in files:
        path = os.path.join(GENERATED_FOLDER, file)
        date = datetime.fromtimestamp(os.path.getmtime(path))
        papers.append((file, date.strftime("%Y-%m-%d %H:%M:%S")))

    return render_template("history.html", papers=papers)


# ---------------- DOWNLOAD ----------------
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(GENERATED_FOLDER, filename, as_attachment=True)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)