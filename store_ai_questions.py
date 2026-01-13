from classifier import classify_sentence
from question_generator import generate_question
from database import insert_question

syllabus = [
    "Operating system is the interface between user and hardware",
    "Difference between process and thread",
    "Explain memory management techniques"
]

for line in syllabus:
    q_type = classify_sentence(line)
    question = generate_question(line, q_type)

    if q_type == "definition":
        marks = 2
    elif q_type == "comparison":
        marks = 5
    elif q_type == "analysis":
        marks = 10
    else:
        marks = 5

    insert_question(question, q_type, marks)

    print("Stored:", question, "|", q_type, "|", marks)
