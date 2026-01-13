from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

text = "Operating system manages CPU scheduling and memory."

input_text = "generate questions: " + text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(input_ids, num_return_sequences=3)

for i, output in enumerate(outputs, 1):
    print(i, tokenizer.decode(output, skip_special_tokens=True))
