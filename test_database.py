from database import insert_question, fetch_questions

# insert sample questions
insert_question("What is control bus?", "definition", 2)
insert_question("Explain DMA transfer.", "analysis", 10)

# fetch and print questions
questions = fetch_questions()
for q in questions:
    print(q)
