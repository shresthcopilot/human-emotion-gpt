from pathlib import Path

from src.tokenizer.tokenizer import CharacterTokenizer


# Load training data
dataset_path = Path("data/raw/dataset.txt")

text = dataset_path.read_text(encoding="utf-8")


# Create tokenizer
tokenizer = CharacterTokenizer(text)


print("=" * 50)
print("CHARACTER TOKENIZER")
print("=" * 50)

print(f"Dataset characters : {len(text)}")
print(f"Vocabulary size    : {tokenizer.vocab_size}")

print("\nVocabulary:")
print(tokenizer.chars)


# Test encoding
sample = "The computer"

encoded = tokenizer.encode(sample)

print("\nOriginal:")
print(sample)

print("\nEncoded:")
print(encoded)


# Test decoding
decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)


# Verify
assert decoded == sample

print("\nTokenizer test PASSED!")