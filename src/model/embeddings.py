import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into learnable vectors.
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Gives each position in the sequence a learnable vector.
    """

    def __init__(self, context_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        return self.embedding(positions)


class InputEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            context_length,
            embedding_dim
        )

    def forward(self, token_ids):
        token_vectors = self.token_embedding(token_ids)

        position_vectors = self.position_embedding(token_ids)

        return token_vectors + position_vectors