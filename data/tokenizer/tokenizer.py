class CharacterTokenizer:

    def __init__(self, text):
        self.chars = sorted(list(set(text)))

        self.char_to_id = {
            char: i
            for i, char in enumerate(self.chars)
        }

        self.id_to_char = {
            i: char
            for i, char in enumerate(self.chars)
        }

        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [
            self.char_to_id[char]
            for char in text
        ]

    def decode(self, token_ids):
        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )