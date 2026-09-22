import torch

from src.tokenizer.tokenizer import CharacterTokenizer
from src.model.gpt import GPTModel
from src.inference.generate import generate_text


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

with open(
    "data/raw/dataset.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# --------------------------------------------------
# Create tokenizer
# --------------------------------------------------

tokenizer = CharacterTokenizer(text)

vocab_size = tokenizer.vocab_size
context_length = 16
embedding_dim = 64
num_heads = 4
hidden_dim = 256
num_layers = 2


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = GPTModel(
    vocab_size=vocab_size,
    context_length=context_length,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "checkpoints/minigpt.pt",
        map_location="cpu"
    )
)

model.eval()


# --------------------------------------------------
# Prompts to test
# --------------------------------------------------

prompts = [
    ".",
    "A",
    "The",
    "A transformer",
    "attention",
    "The computer",
    "computer",
    "transformer",
    "program",
    "information"
]


# --------------------------------------------------
# Generate text for every prompt
# --------------------------------------------------

print("=" * 70)
print("MINIGPT MULTIPLE PROMPT TEST")
print("=" * 70)

for prompt in prompts:

    try:

        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            context_length=context_length,
            temperature=0.7,
            top_k=5
        )

        print("\n" + "-" * 70)
        print(f"PROMPT: {prompt}")
        print("-" * 70)
        print(generated_text)

    except KeyError as error:

        print("\n" + "-" * 70)
        print(f"PROMPT: {prompt}")
        print("-" * 70)
        print(
            f"Skipped: character {error} "
            "is not in the vocabulary."
        )

