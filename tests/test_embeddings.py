import torch

from src.tokenizer.tokenizer import CharacterTokenizer
from src.model.embeddings import InputEmbedding


# Load dataset
text = open(
    "data/raw/dataset.txt",
    encoding="utf-8"
).read()


# Tokenizer
tokenizer = CharacterTokenizer(text)


# Model settings
vocab_size = tokenizer.vocab_size
context_length = 16
embedding_dim = 64


# Create embedding layer
embedding = InputEmbedding(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=embedding_dim
)


# Example text
sample = "The computer"

token_ids = torch.tensor([
    tokenizer.encode(sample)
])


# Convert tokens to vectors
embedded = embedding(token_ids)


print("=" * 50)
print("TOKEN + POSITION EMBEDDING")
print("=" * 50)

print(f"Vocabulary size : {vocab_size}")
print(f"Context length  : {context_length}")
print(f"Embedding size  : {embedding_dim}")

print("\nInput:")
print(sample)

print("\nToken IDs:")
print(token_ids)

print("\nOutput shape:")
print(embedded.shape)

print("\nFirst token vector:")
print(embedded[0, 0])

print("\nEmbedding test PASSED!")