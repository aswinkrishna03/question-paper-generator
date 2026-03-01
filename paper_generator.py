from database import fetch_questions

def generate_question_paper():

    questions = fetch_questions()

    short_questions = []
    long_questions = []

    for q, q_type, marks in questions:
        if marks == 3:
            short_questions.append(q)
        else:
            long_questions.append(q)

    # Ensure 5 short questions
    if len(short_questions) < 5:
        needed = 5 - len(short_questions)
        short_questions.extend(long_questions[:needed])
        long_questions = long_questions[needed:]

    # Ensure 2 long questions
    if len(long_questions) < 2:
        needed = 2 - len(long_questions)
        long_questions.extend(short_questions[:needed])

    paper = []

    paper.append("AI QUESTION PAPER")
    paper.append("Duration: 2 Hours")
    paper.append("Total Marks: 35")
    paper.append("=========================================")
    paper.append("")

    paper.append("SECTION A")
    paper.append("Answer ALL questions (5 × 3 = 15 Marks)")
    paper.append("")

    for i, q in enumerate(short_questions[:5], 1):
        paper.append(f"{i}. {q} (3 Marks)")
        paper.append("")

    paper.append("SECTION B")
    paper.append("Answer ALL questions (2 × 10 = 20 Marks)")
    paper.append("")

    for i, q in enumerate(long_questions[:2], 1):
        paper.append(f"{i}. {q} (10 Marks)")
        paper.append("")

    return paper