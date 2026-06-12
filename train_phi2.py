from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
 
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
 
model = AutoModelForCausalLM.from_pretrained(model_name)
 
lora_config = LoraConfig(
   r=8,
   lora_alpha=16,
   lora_dropout=0.05,
  bias="none",
  task_type="CAUSAL_LM"
)
 
model = get_peft_model(model, lora_config)
 
dataset = load_dataset("json", data_files="expanded_dataset.jsonl")["train"]
 
def tokenize(example):
   text = f"Instruction: {example['instruction']}\nAnswer: {example['output']}"
   return tokenizer(
       text,
      truncation=True,
      padding="max_length",
       max_length=128
   )
 
tokenized = dataset.map(tokenize)
 
training_args = TrainingArguments(
  output_dir="./tinyllama-coding-model",
  num_train_epochs=1,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=2,
   logging_steps=1,
   save_steps=20,
  learning_rate=2e-4,
  report_to="none"
)
 
trainer = Trainer(
   model=model,
  args=training_args,
  train_dataset=tokenized,
  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
 
trainer.train()
 
model.save_pretrained("./tinyllama-coding-model")
tokenizer.save_pretrained("./tinyllama-coding-model")