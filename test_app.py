import os
import app
from flask import Flask
import logging
logging.basicConfig(level=logging.INFO)

app.app.config['TESTING'] = True
client = app.app.test_client()
with client.session_transaction() as sess:
    sess["examiner_id"] = 1
    sess["examiner_username"] = "test"

import question_generator

def mock_gen(*args, **kwargs):
    output = "SECTION A\n"
    for i in range(1, 11):
        md = (i % 5) + 1
        output += f"{i}. [Module: {md}] [Bloom: Apply] This is mock question number {i} for module {md}.\n"
    output += "\nSECTION B\n"
    for i in range(11, 16):
        md = ((i - 11) % 5) + 1
        output += f"{i}. [Module: {md}] [Bloom: Analyze] This is mock question number {i} for module {md}.\n"
    return output

question_generator.generate_paper_gemini = mock_gen
app._gemini_generation_enabled = lambda: True

with app.app.app_context():
    res = client.post("/", data={
        "pdfs": [
            (open("uploads/OOP JAVA -M1 Ktunotes.in.pdf", "rb"), "OOP JAVA -M1 Ktunotes.in.pdf"),
            (open("uploads/OOP JAVA -M2 Ktunotes.in.pdf", "rb"), "OOP JAVA -M2 Ktunotes.in.pdf"),
            (open("uploads/OOP JAVA -M3 Ktunotes.in.pdf", "rb"), "OOP JAVA -M3 Ktunotes.in.pdf"),
            (open("uploads/OOP JAVA -M4 Ktunotes.in.pdf", "rb"), "OOP JAVA -M4 Ktunotes.in.pdf"),
            (open("uploads/OOP JAVA -M5 Ktunotes.in.pdf", "rb"), "OOP JAVA -M5 Ktunotes.in.pdf")
        ],
        "blooms_levels": ["Remember", "Understand"],
        "blooms_distribution": "balanced",
        "paper_model": "100",
        "paper_title": "Test Title"
    }, content_type='multipart/form-data')
    print(res.status_code)
    text_data = res.get_data(as_text=True)
    if "Question generation did not satisfy" in text_data:
        print("FAILED: Question generation failed pattern.")
        print(text_data[:2000])
    elif "Insufficient valid concepts extracted from" in text_data:
        print("FAILED CONCEPTS:")
        print(text_data[:500])
    else:
        print("SUCCESS")
