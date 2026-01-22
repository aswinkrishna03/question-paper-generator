from database import fetch_questions

def generate_question_paper():
    questions = fetch_questions()

    paper = []
    paper.append("AUTOMATIC QUESTION PAPER\n")

    qno = 1
    for q in questions:
        paper.append(f"{qno}. {q[0]}")
        qno += 1

    return paper
