from database import fetch_questions

def generate_question_paper():
    questions = fetch_questions()

    paper = "AUTOMATIC QUESTION PAPER\n\n"

    if not questions:
        paper += "No questions generated.\n"
        return paper

    for i, q in enumerate(questions, start=1):
        paper += f"{i}. {q[0]}\n"

    return paper
