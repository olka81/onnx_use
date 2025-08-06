# src/download_tokenizer.py

from transformers import GPT2TokenizerFast
from pathlib import Path

# Папка, куда сохраняем
output_dir = Path(__file__).resolve().parent.parent / "third_party" / "gpt2_tokenizer"
output_dir.mkdir(parents=True, exist_ok=True)

# Загрузка и сохранение токенизатора
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.save_pretrained(str(output_dir))

print(f"Tokenizer saved to: {output_dir}")