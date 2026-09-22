import os
import torch

from src.config import (
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    NUM_HEADS,
    HIDDEN_DIM,
    NUM_LAYERS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    STEPS_PER_EPOCH,
    EVAL_INTERVAL,
    EVAL_BATCHES,
    DEVICE
)

from src.tokenizer.tokenizer import CharacterTokenizer
from src.training.dataset import RandomTextBatcher
from src.training.loss import LanguageModelLoss
from src.model.gpt import GPTModel


# ============================================================
# LOAD DATA
# ============================================================

with open(
    "data/processed/train.txt",
    "r",
    encoding="utf-8"
) as file:
    train_text = file.read()


with open(
    "data/processed/val.txt",
    "r",
    encoding="utf-8"
) as file:
    val_text = file.read()


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print(f"Training characters   : {len(train_text):,}")
print(f"Validation characters : {len(val_text):,}")


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = CharacterTokenizer(text=train_text)

vocab_size = tokenizer.vocab_size

print(f"Vocabulary size       : {vocab_size}")


# ============================================================
# ENCODE DATA
# ============================================================

train_token_ids = tokenizer.encode(train_text)
val_token_ids = tokenizer.encode(val_text)


print(f"Training tokens       : {len(train_token_ids):,}")
print(f"Validation tokens     : {len(val_token_ids):,}")


# ============================================================
# BATCHERS
# ============================================================

train_batcher = RandomTextBatcher(
    token_ids=train_token_ids,
    context_length=CONTEXT_LENGTH,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

val_batcher = RandomTextBatcher(
    token_ids=val_token_ids,
    context_length=CONTEXT_LENGTH,
    batch_size=BATCH_SIZE,
    device=DEVICE
)


# ============================================================
# MODEL
# ============================================================

model = GPTModel(
    vocab_size=vocab_size,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

model.to(DEVICE)


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)

print(f"Parameters          : {total_parameters:,}")
print(f"Context length      : {CONTEXT_LENGTH}")
print(f"Embedding dimension : {EMBEDDING_DIM}")
print(f"Attention heads     : {NUM_HEADS}")
print(f"Hidden dimension    : {HIDDEN_DIM}")
print(f"Transformer layers  : {NUM_LAYERS}")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Device              : {DEVICE}")


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

loss_function = LanguageModelLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(batch_sampler):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for _ in range(EVAL_BATCHES):

            x, y = batch_sampler.get_batch()

            logits = model(x)

            loss = loss_function(
                logits,
                y
            )

            total_loss += loss.item()

    model.train()

    return total_loss / EVAL_BATCHES


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MINIGPT")
print("=" * 70)

global_step = 0

best_val_loss = float("inf")

train_losses = []
val_losses = []


for epoch in range(NUM_EPOCHS):

    model.train()

    epoch_loss = 0.0

    for step in range(STEPS_PER_EPOCH):

        # ----------------------------------------------------
        # GET RANDOM BATCH
        # ----------------------------------------------------

        x, y = train_batcher.get_batch()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        logits = model(x)


        # ----------------------------------------------------
        # CALCULATE LOSS
        # ----------------------------------------------------

        loss = loss_function(
            logits,
            y
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # TRACK LOSS
        # ----------------------------------------------------

        epoch_loss += loss.item()

        global_step += 1


        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        if global_step % EVAL_INTERVAL == 0:

            train_loss = epoch_loss / (
                step + 1
            )

            val_loss = evaluate(
                val_batcher
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"Step {global_step:06d} "
                f"| Epoch {epoch + 1}/{NUM_EPOCHS} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f}"
            )


    # ========================================================
    # EPOCH COMPLETE
    # ========================================================

    average_train_loss = epoch_loss / STEPS_PER_EPOCH

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS} complete "
        f"| Train Loss: {average_train_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    current_val_loss = evaluate(
        val_batcher
    )

    if current_val_loss < best_val_loss:

        best_val_loss = current_val_loss

        os.makedirs(
            "checkpoints",
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            "checkpoints/minigpt_best.pt"
        )

        print(
            f"Best model saved "
            f"| Val Loss: {best_val_loss:.4f}"
        )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    "checkpoints/minigpt_wikitext.pt"
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Final model:"
)

print(
    "checkpoints/minigpt_wikitext.pt"
)

print(
    "Best model:"
)

print(
    "checkpoints/minigpt_best.pt"
)