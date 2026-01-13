from classifier import classify_sentence
from question_generator import generate_question

syllabus = [
    "Operating system is the interface between user and hardware",
    "Difference between process and thread",
    "Explain memory management techniques"
]

for line in syllabus:
    q_type = classify_sentence(line)
    question = generate_question(line, q_type)

    print("Syllabus :", line)
    print("Type     :", q_type)
    print("Question :", question)
    print("-" * 50)
