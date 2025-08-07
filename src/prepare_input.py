from transformers import GPT2TokenizerFast
import sys

if len(sys.argv) < 2:
    print("Usage: python prepare_input.py \"Your prompt here\"")
    exit(1)

prompt = sys.argv[1]

tokenizer = GPT2TokenizerFast.from_pretrained(
    "./third_party/gpt2_tokenizer",
    local_files_only=True
)

input_ids = tokenizer.encode(prompt, add_special_tokens=False)

# Save to text file as space-separated integers
with open("input_ids.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, input_ids)))

print("Saved input_ids.txt:", input_ids)
