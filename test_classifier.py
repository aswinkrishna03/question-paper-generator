from classifier import classify_sentence

lines = [
    "Operating system is the interface between user and hardware",
    "Difference between process and thread",
    "Explain memory management techniques"
]

for line in lines:
    print(line, "->", classify_sentence(line))
