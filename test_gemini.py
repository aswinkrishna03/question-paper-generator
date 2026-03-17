import os
import sys

# Set a dummy key so the client initializes
os.environ["GEMINI_API_KEY"] = "dummy_key_for_testing"

from gemini_validator import generate_questions_with_gemini, validate_questions_with_gemini, rewrite_questions_with_gemini

print("Testing generate_questions_with_gemini...")
res = generate_questions_with_gemini("Hello, this is a test prompt.")
print("Result:")
print(res)

print("Testing validate_questions_with_gemini...")
res2 = validate_questions_with_gemini(["What is testing?"])
print("Result:")
print(res2)
