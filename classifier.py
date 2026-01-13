def classify_sentence(sentence):
    sentence = sentence.lower()

    if any(word in sentence for word in ["define", "defined", "refers", "is", "are"]):
        return "definition"

    if any(word in sentence for word in ["difference", "compare", "vs", "between"]):
        return "comparison"

    if any(word in sentence for word in ["explain", "analyze", "discuss", "describe"]):
        return "analysis"

    return "general"
