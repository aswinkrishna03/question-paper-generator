from transformers import T5Tokenizer, T5ForConditionalGeneration

# Global variables (initially empty)
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Loading T5 model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        print("T5 model loaded.")

def generate_question(text, q_type):
    load_model()

    if q_type == "definition":
        prompt = f"Generate a definition question from this text:\n{text}"
    elif q_type == "comparison":
        prompt = f"Generate a comparison question from this text:\n{text}"
    elif q_type == "analysis":
        prompt = f"Generate an analytical question from this text:\n{text}"
    else:
        prompt = f"Generate a question from this text:\n{text}"

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    output = model.generate(
        input_ids,
        max_length=60,
        num_beams=4,
        early_stopping=True
    )

    question = tokenizer.decode(output[0], skip_special_tokens=True)

    # Safety fallback
    if len(question.split()) < 4:
        if q_type == "definition":
            question = f"What is {text}?"
        elif q_type == "comparison":
            question = f"Compare {text}."
        else:
            question = f"Explain {text}."

    return question
