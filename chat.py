from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./tinyllama-coding-model"

print("Loading model... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

print("Coding Assistant Ready!")
print("Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        print("Goodbye!")
        break

    text = f"""You are a professional coding assistant.
Give correct, beginner-friendly programming answers.
Provide complete code when asked.

Instruction: {prompt}

Answer:"""

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in result:
        answer = result.split("Answer:", 1)[1].strip()
    else:
        answer = result.strip()

    print("\nAI:")
    print(answer)
    print("-" * 50)