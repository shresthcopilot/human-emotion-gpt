import torch

from src.model.transformer_block import TransformerBlock


# Model settings
embedding_dim = 64
num_heads = 4
context_length = 16
hidden_dim = 256


# Create Transformer block
transformer_block = TransformerBlock(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    context_length=context_length,
    hidden_dim=hidden_dim
)


# Fake input
x = torch.randn(
    1,
    12,
    embedding_dim
)


# Forward pass
output = transformer_block(x)


print("=" * 50)
print("TRANSFORMER BLOCK")
print("=" * 50)

print(f"Input shape  : {x.shape}")
print(f"Output shape : {output.shape}")

print(f"\nEmbedding dimension : {embedding_dim}")
print(f"Number of heads     : {num_heads}")
print(f"Head dimension      : {embedding_dim // num_heads}")
print(f"Hidden dimension    : {hidden_dim}")

print("\nTransformer block output:")
print(output[0, 0])

assert output.shape == x.shape

print("\nTransformer block test PASSED!")