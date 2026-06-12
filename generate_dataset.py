import json

samples = [
    {
        "topic": "factorial",
        "questions": [
            "Write Python program for factorial",
            "Create factorial program in Python",
            "Find factorial using loop in Python",
            "Factorial using recursion in Python"
        ],
        "answer": """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(factorial(5))"""
    },
    {
        "topic": "palindrome",
        "questions": [
            "Write Python palindrome program",
            "Check palindrome in Python",
            "Python string palindrome example"
        ],
        "answer": """text = "madam"
print(text == text[::-1])"""
    }
]

with open("expanded_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in samples:
        for q in item["questions"]:
            row = {
                "instruction": q,
                "input": "",
                "output": item["answer"]
            }
            f.write(json.dumps(row) + "\n")

print("Dataset Generated Successfully")