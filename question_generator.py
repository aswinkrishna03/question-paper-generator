import re
import random


def extract_concept(paragraph):
    """
    Extract meaningful concept from paragraph.
    """

    # Remove unwanted patterns
    paragraph = re.sub(r'\b(module|unit|chapter|figure|diagram)\b', '', paragraph, flags=re.IGNORECASE)

    words = paragraph.split()

    # Take first meaningful 3–5 words
    concept = " ".join(words[:5])

    concept = concept.strip(" ,.-")

    return concept


def generate_short_question(concept):
    """
    3-mark questions (definition-level)
    """

    templates = [
        f"What is {concept}?",
        f"Define {concept}.",
        f"List the types of {concept}.",
        f"Give examples of {concept}."
    ]

    return random.choice(templates)


def generate_long_question(concept):
    """
    10-mark analytical questions
    """

    templates = [
        f"Explain the concept of {concept} in detail.",
        f"Discuss the advantages and limitations of {concept}.",
        f"Analyse the impact of {concept} on environmental sustainability.",
        f"Evaluate the applications of {concept} in modern industrial systems."
    ]

    return random.choice(templates)


def generate_question(paragraph, q_type):

    if len(paragraph.split()) < 20:
        return None

    concept = extract_concept(paragraph)

    if not concept or len(concept.split()) < 2:
        return None

    # Return both types — marks decided in app.py
    return concept