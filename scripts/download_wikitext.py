from datasets import load_dataset
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("data/raw/wikitext")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Download WikiText-2 RAW
# --------------------------------------------------

print("=" * 60)
print("DOWNLOADING WIKITEXT-2")
print("=" * 60)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1"
)


# --------------------------------------------------
# Save each split as a text file
# --------------------------------------------------

splits = {
    "train": "wiki.train.raw",
    "validation": "wiki.valid.raw",
    "test": "wiki.test.raw"
}


for split_name, filename in splits.items():

    output_path = OUTPUT_DIR / filename

    print(f"\nSaving {split_name}...")

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for row in dataset[split_name]:

            file.write(row["text"])
            file.write("\n")

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

for filename in splits.values():
    print(f"data/raw/wikitext/{filename}")