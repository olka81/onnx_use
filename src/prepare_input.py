from transformers import GPT2TokenizerFast
import sys

if len(sys.argv) < 2:
    print("Usage: python prepare_input.py <digit>")
    exit(1)

digit = sys.argv[1]
prompt = f"Do you know {digit}?"

tokenizer = GPT2TokenizerFast.from_pretrained(
    "./third_party/gpt2_tokenizer",
    local_files_only=True
)

encoded = tokenizer(prompt, return_attention_mask=True, add_special_tokens=False)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

# Save input_ids
with open("input_ids.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, input_ids)))

# Save attention_mask
with open("attention_mask.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, attention_mask)))

print("Saved input_ids.txt:", input_ids)
print("Saved attention_mask.txt:", attention_mask)