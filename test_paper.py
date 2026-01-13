from paper_generator import generate_question_paper

paper = generate_question_paper()

print("QUESTION PAPER\n")

for i, q in enumerate(paper, 1):
    print(f"{i}. {q}")
