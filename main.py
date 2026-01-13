from paper_generator import generate_question_paper
from pdf_generator import generate_pdf

paper = generate_question_paper()
generate_pdf(paper)

print("Question paper PDF generated successfully.")
