from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from pypdf import PdfReader
import os
import re

from classifier import classify_sentence
from question_generator import generate_question
from database import (
    insert_question,
    clear_questions,
    init_db,
    create_user,
    validate_user,
    save_paper,
    get_user_papers
)
from paper_generator import generate_question_paper
from pdf_generator import generate_pdf

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

# Initialize database
init_db()

UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "generated_papers"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


# ===============================
# LOGIN
# ===============================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user_id = validate_user(username, password)

        if user_id:
            session["user_id"] = user_id
            session["username"] = username
            return redirect(url_for("upload_pdf"))
        else:
            return "Invalid credentials"

    return render_template("login.html")


# ===============================
# REGISTER
# ===============================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        create_user(
            request.form["username"],
            request.form["password"]
        )
        return redirect(url_for("login"))

    return render_template("register.html")


# ===============================
# LOGOUT
# ===============================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ===============================
# MAIN PAGE (PROTECTED)
# ===============================
@app.route("/", methods=["GET", "POST"])
def upload_pdf():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        clear_questions()
        seen_topics = set()

        pdf_file = request.files["pdf"]

        if pdf_file.filename == "":
            return "No file selected"

        file_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
        pdf_file.save(file_path)

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        # Clean text
        text = text.replace("\n", " ")
        text = re.sub(r"[■•–]", " ", text)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

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

                if len(topic) < 6:
                    continue

                if re.search(r"[A-Z]{3,}\d{2,}", topic):
                    continue

                if topic.isupper():
                    continue

                if topic_lower in seen_topics:
                    continue
                seen_topics.add(topic_lower)

                if topic_lower in ["resources", "engineering", "syllabus"]:
                    continue

                q_type = classify_sentence(topic)
                question = generate_question(topic, q_type)

                marks = 2 if q_type == "definition" else 5
                insert_question(question, q_type, marks)

                count += 1
                if count >= 10:
                    break

            if count >= 10:
                break

        # Generate paper
        paper = generate_question_paper()
        filepath = generate_pdf(paper)
        filename = os.path.basename(filepath)

        # Save history
        save_paper(session["user_id"], filename)

        return redirect(url_for("result", file=filename))

    return render_template("upload.html", username=session["username"])


# ===============================
# RESULT PAGE
# ===============================
@app.route("/result")
def result():
    if "user_id" not in session:
        return redirect(url_for("login"))

    file_name = request.args.get("file")
    return render_template("result.html", file_name=file_name)


# ===============================
# DOWNLOAD
# ===============================
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(
        directory=GENERATED_FOLDER,
        path=filename,
        as_attachment=True
    )


# ===============================
# HISTORY
# ===============================
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    papers = get_user_papers(session["user_id"])
    return render_template("history.html", papers=papers)


# ===============================
# RUN SERVER (RENDER READY)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)