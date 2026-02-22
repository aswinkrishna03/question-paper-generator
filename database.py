import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_NAME = "questions.db"
# ===============================
# FETCH QUESTIONS (USED BY PAPER GENERATOR)
# ===============================

def fetch_questions():
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT question, type, marks FROM questions")
    rows = c.fetchall()

    conn.close()
    return rows


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)

    # GENERATED PAPERS TABLE
    c.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        created_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # QUESTIONS TABLE (YOUR ORIGINAL SYSTEM)
    c.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        type TEXT,
        marks INTEGER
    )
    """)

    conn.commit()
    conn.close()


# ===============================
# QUESTION FUNCTIONS (ORIGINAL)
# ===============================

def insert_question(question, q_type, marks):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO questions (question, type, marks) VALUES (?, ?, ?)",
        (question, q_type, marks)
    )
    conn.commit()
    conn.close()


def clear_questions():
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM questions")
    conn.commit()
    conn.close()


# ===============================
# USER AUTH FUNCTIONS
# ===============================

def create_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    hashed = generate_password_hash(password)

    c.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, hashed)
    )

    conn.commit()
    conn.close()


def validate_user(username, password):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "SELECT id, password FROM users WHERE username = ?",
        (username,)
    )
    user = c.fetchone()
    conn.close()

    if user and check_password_hash(user[1], password):
        return user[0]
    return None


# ===============================
# PAPER HISTORY FUNCTIONS
# ===============================

def save_paper(user_id, filename):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "INSERT INTO papers (user_id, filename, created_at) VALUES (?, ?, ?)",
        (user_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )

    conn.commit()
    conn.close()


def get_user_papers(user_id):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        "SELECT filename, created_at FROM papers WHERE user_id = ? ORDER BY id DESC",
        (user_id,)
    )

    papers = c.fetchall()
    conn.close()
    return papers