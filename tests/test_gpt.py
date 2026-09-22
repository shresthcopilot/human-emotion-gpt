import torch

from src.tokenizer.tokenizer import CharacterTokenizer
from src.model.gpt import GPTModel


# --------------------------------
# Load dataset
# --------------------------------

with open(
    "data/raw/dataset.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# --------------------------------
# Create tokenizer
# --------------------------------

tokenizer = CharacterTokenizer(text)


# --------------------------------
# Model settings
# --------------------------------

vocab_size = tokenizer.vocab_size
context_length = 16
embedding_dim = 64
num_heads = 4
hidden_dim = 256
num_layers = 2


# --------------------------------
# Create model
# --------------------------------

model = GPTModel(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)


# --------------------------------
# Prepare input
# --------------------------------

sample_text = "The computer"

token_ids = tokenizer.encode(
    sample_text
)

input_tensor = torch.tensor(
    [token_ids],
    dtype=torch.long
)


# --------------------------------
# Forward pass
# --------------------------------

logits = model(input_tensor)


# --------------------------------
# Print information
# --------------------------------

print("=" * 50)
print("GPT MODEL")
print("=" * 50)

print(f"Vocabulary size     : {vocab_size}")
print(f"Context length      : {context_length}")
print(f"Embedding dimension : {embedding_dim}")
print(f"Number of heads     : {num_heads}")
print(f"Hidden dimension    : {hidden_dim}")
print(f"Number of layers    : {num_layers}")

print("\nInput:")
print(sample_text)

print("\nToken IDs:")
print(token_ids)

print("\nInput tensor shape:")
print(input_tensor.shape)

print("\nLogits shape:")
print(logits.shape)

print("\nLogits for first token:")
print(logits[0, 0])

assert logits.shape == (
    1,
    len(token_ids),
    vocab_size
)

print("\nGPT model test PASSED!")