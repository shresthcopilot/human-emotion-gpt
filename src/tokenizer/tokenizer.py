class CharacterTokenizer:
    """
    Character-level tokenizer with an <UNK> token.

    The vocabulary is created from training data only.

    Characters that are not present in the training
    vocabulary are mapped to <UNK>.
    """

    UNK_TOKEN = "<UNK>"

    def __init__(self, text=None, chars=None):

        if chars is not None:
            self.chars = sorted(chars)

        elif text is not None:
            self.chars = sorted(set(text))

        else:
            raise ValueError(
                "Provide either text or chars."
            )

        # Add special unknown token
        if self.UNK_TOKEN not in self.chars:
            self.chars.append(self.UNK_TOKEN)

        self.vocab_size = len(self.chars)

        self.char_to_id = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.id_to_char = {
            index: char
            for index, char in enumerate(self.chars)
        }

        self.unk_id = self.char_to_id[
            self.UNK_TOKEN
        ]

    def encode(self, text):
        """
        Convert text into token IDs.

        Unknown characters become <UNK>.
        """

        return [
            self.char_to_id.get(
                char,
                self.unk_id
            )
            for char in text
        ]

    def decode(self, token_ids):
        """
        Convert token IDs back into text.
        """

        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )

    def __len__(self):
        return self.vocab_size