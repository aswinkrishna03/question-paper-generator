import sqlite3

DB_NAME = "questions.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            q_type TEXT,
            marks INTEGER
        )
    """)
    conn.commit()
    conn.close()

def clear_questions():
    create_table()  # <-- THIS IS THE FIX
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM questions")
    conn.commit()
    conn.close()

def insert_question(question, q_type, marks):
    create_table()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO questions (question, q_type, marks) VALUES (?, ?, ?)",
        (question, q_type, marks)
    )
    conn.commit()
    conn.close()

def fetch_questions():
    create_table()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT question FROM questions")
    rows = cursor.fetchall()
    conn.close()
    return rows
