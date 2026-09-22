from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_DIR = Path("data/raw/wikitext")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Read raw WikiText files
# --------------------------------------------------

train_path = RAW_DIR / "wiki.train.raw"
valid_path = RAW_DIR / "wiki.valid.raw"
test_path = RAW_DIR / "wiki.test.raw"


train_text = train_path.read_text(
    encoding="utf-8"
)

valid_text = valid_path.read_text(
    encoding="utf-8"
)

test_text = test_path.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# Basic cleaning
# --------------------------------------------------

def clean_text(text):
    """
    Perform minimal cleaning.

    We intentionally do NOT aggressively clean
    punctuation, capitalization, or numbers because
    this is a language-modeling dataset.
    """

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")

    # Remove trailing spaces from lines
    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    return text


train_text = clean_text(train_text)
valid_text = clean_text(valid_text)
test_text = clean_text(test_text)


# --------------------------------------------------
# Save processed datasets
# --------------------------------------------------

(PROCESSED_DIR / "train.txt").write_text(
    train_text,
    encoding="utf-8"
)

(PROCESSED_DIR / "val.txt").write_text(
    valid_text,
    encoding="utf-8"
)

(PROCESSED_DIR / "test.txt").write_text(
    test_text,
    encoding="utf-8"
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("=" * 60)
print("WIKITEXT-2 PREPROCESSING")
print("=" * 60)

print(f"Training characters   : {len(train_text):,}")
print(f"Validation characters : {len(valid_text):,}")
print(f"Test characters       : {len(test_text):,}")

print("\nSaved:")
print("data/processed/train.txt")
print("data/processed/val.txt")
print("data/processed/test.txt")