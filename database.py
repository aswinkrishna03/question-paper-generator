import sqlite3

def create_table():
    conn = sqlite3.connect("questions.db")
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


def insert_question(question, q_type, marks):
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO questions (question, q_type, marks)
        VALUES (?, ?, ?)
    """, (question, q_type, marks))

    conn.commit()
    conn.close()


def fetch_questions():
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT question, q_type, marks FROM questions")
    rows = cursor.fetchall()

    conn.close()
    return rows


def clear_questions():
    conn = sqlite3.connect("questions.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM questions")

    conn.commit()
    conn.close()


create_table()