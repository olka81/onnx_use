import sys
from transformers import GPT2TokenizerFast

if len(sys.argv) < 2:
    print("Usage: python decode_output.py <id1> <id2> ...")
    sys.exit(1)

# Загружаем токенизатор
tokenizer = GPT2TokenizerFast.from_pretrained("./third_party/gpt2_tokenizer")

# Преобразуем аргументы в список ID
token_ids = list(map(int, sys.argv[1:]))

# Декодируем
text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(text)