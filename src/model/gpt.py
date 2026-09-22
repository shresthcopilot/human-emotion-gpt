import torch
import torch.nn as nn

from src.model.embeddings import InputEmbedding
from src.model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    A tiny GPT-style language model.
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_layers
    ):
        super().__init__()

        # Token + positional embeddings
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim
        )

        # Stack multiple Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    hidden_dim=hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        # Convert model representation
        # into vocabulary logits
        self.output_projection = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):
        """
        token_ids shape:

            [batch_size, sequence_length]

        Returns:

            [batch_size, sequence_length, vocab_size]
        """

        # --------------------------------
        # 1. Token + positional embeddings
        # --------------------------------

        x = self.embedding(token_ids)

        # --------------------------------
        # 2. Transformer blocks
        # --------------------------------

        for block in self.transformer_blocks:
            x = block(x)

        # --------------------------------
        # 3. Final normalization
        # --------------------------------

        x = self.final_layer_norm(x)

        # --------------------------------
        # 4. Convert to vocabulary logits
        # --------------------------------

        logits = self.output_projection(x)

        return logits