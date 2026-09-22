import torch

from src.tokenizer.tokenizer import CharacterTokenizer
from src.model.embeddings import InputEmbedding
from src.model.attention import MultiHeadAttention


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


# Create input embedding
embedding = InputEmbedding(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=embedding_dim
)


# Create attention
num_heads = 4

attention = MultiHeadAttention(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    context_length=context_length
)


# Example input
sample = "The computer"

token_ids = torch.tensor([
    tokenizer.encode(sample)
])


# Convert tokens to embeddings
x = embedding(token_ids)


# Apply self-attention
output= attention(x)


print("=" * 50)
print("MULTI-HEAD SELF-ATTENTION")
print("=" * 50)

print(f"Input shape  : {x.shape}")
print(f"Output shape : {output.shape}")

print(f"\nNumber of heads : {num_heads}")
print(f"Head dimension  : {embedding_dim // num_heads}")

print("\nInput:")
print(sample)

print("\nMulti-head attention output:")
print(output[0, 0])

print("\nMulti-head attention test PASSED!")