from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

tokenizer = None
model = None
device = None


def load_model():
    global tokenizer, model, device

    if tokenizer is None:
        print("Loading model...")

        tokenizer = T5Tokenizer.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )

        model = T5ForConditionalGeneration.from_pretrained(
            "mrm8488/t5-base-finetuned-question-generation-ap"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        print("Model loaded.")


def clean_output(question):
    question = question.strip()
    question = re.sub(r"\s+", " ", question)

    if not question.endswith("?"):
        question = question.rstrip(".") + "?"

    return question


def is_weak_question(question):
    weak_patterns = [
        "what is shown",
        "what is an example",
        "what other",
        "what does",
        "what is the benefit",
        "what can be addressed",
        "where can",
    ]

    q = question.lower()
    return any(pattern in q for pattern in weak_patterns)


def generate_question(paragraph, q_type):

    load_model()

    paragraph = paragraph.strip()

    # Limit long paragraphs
    if len(paragraph.split()) > 150:
        paragraph = " ".join(paragraph.split()[:150])

    input_text = f"context: {paragraph}"

    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    output = model.generate(
        input_ids,
        max_length=64,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    question = tokenizer.decode(output[0], skip_special_tokens=True)
    question = clean_output(question)

    # 🔥 Instead of rejecting weak question, REWRITE it
    if is_weak_question(question):
        return "Explain the key concepts discussed in the paragraph."

    # Fallback if too short
    if len(question.split()) < 6:
        return "Discuss the important concepts described in the paragraph."

    return question