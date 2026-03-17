def validate_short(q):
    return q.lower().startswith(("what", "define", "list", "give"))

def validate_long(q):
    return q.startswith(("Explain", "Discuss", "Analyse", "Analyze", "Evaluate", "Describe"))