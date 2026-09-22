import torch

from src.model.feed_forward import FeedForward


# Model settings
embedding_dim = 64
hidden_dim = 256


# Create feed-forward network
feed_forward = FeedForward(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
)


# Fake input
x = torch.randn(
    1,
    12,
    embedding_dim
)


# Forward pass
output = feed_forward(x)


print("=" * 50)
print("FEED-FORWARD NETWORK")
print("=" * 50)

print(f"Input shape  : {x.shape}")
print(f"Output shape : {output.shape}")

print(f"\nEmbedding dimension : {embedding_dim}")
print(f"Hidden dimension    : {hidden_dim}")

print("\nFeed-forward output:")
print(output[0, 0])

print("\nFeed-forward test PASSED!")