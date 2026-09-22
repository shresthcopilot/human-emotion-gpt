from pathlib import Path

from src.tokenizer.tokenizer import CharacterTokenizer
from src.training.dataset import TextDataset


# Load dataset
dataset_path = Path("data/raw/dataset.txt")
text = dataset_path.read_text(encoding="utf-8")


# Create tokenizer
tokenizer = CharacterTokenizer(text)

# Convert entire text into token IDs
token_ids = tokenizer.encode(text)


# Create training dataset
context_length = 16

dataset = TextDataset(
    token_ids=token_ids,
    context_length=context_length
)


print("=" * 50)
print("TEXT DATASET")
print("=" * 50)

print(f"Total tokens    : {len(token_ids)}")
print(f"Context length  : {context_length}")
print(f"Training samples: {len(dataset)}")


# Get first training example
x, y = dataset[0]


print("\nInput tokens:")
print(x.tolist())

print("\nTarget tokens:")
print(y.tolist())


# Decode them so we can understand them
print("\nInput text:")
print(tokenizer.decode(x.tolist()))

print("\nTarget text:")
print(tokenizer.decode(y.tolist()))


# Verify the shift
assert x[1:].tolist() == y[:-1].tolist()

print("\nDataset test PASSED!")