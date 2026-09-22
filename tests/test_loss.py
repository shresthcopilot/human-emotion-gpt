import torch

from src.tokenizer.tokenizer import CharacterTokenizer
from src.training.dataset import TextDataset
from src.model.gpt import GPTModel
from src.training.loss import LanguageModelLoss


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
# Tokenizer
# --------------------------------

tokenizer = CharacterTokenizer(text)


# --------------------------------
# Dataset
# --------------------------------

context_length = 16

dataset = TextDataset(
    tokenizer.encode(text),
    context_length
)


# --------------------------------
# Get one training sample
# --------------------------------

x, y = dataset[0]

# Add batch dimension
x = x.unsqueeze(0)
y = y.unsqueeze(0)


# --------------------------------
# Model settings
# --------------------------------

vocab_size = tokenizer.vocab_size
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
# Forward pass
# --------------------------------

logits = model(x)


# --------------------------------
# Calculate loss
# --------------------------------

loss_function = LanguageModelLoss()

loss = loss_function(
    logits,
    y
)


# --------------------------------
# Print results
# --------------------------------

print("=" * 50)
print("LANGUAGE MODEL LOSS")
print("=" * 50)

print(f"Input shape  : {x.shape}")
print(f"Target shape : {y.shape}")
print(f"Logits shape : {logits.shape}")

print(f"\nLoss: {loss.item():.4f}")

print("\nLoss test PASSED!")