import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Create Q, K and V projections
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Final projection after combining heads
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Causal mask
        mask = torch.tril(
            torch.ones(
                context_length,
                context_length
            )
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):
        """
        x:
            [batch_size, sequence_length, embedding_dim]

        returns:
            [batch_size, sequence_length, embedding_dim]
        """

        batch_size, sequence_length, _ = x.shape

        # --------------------------------------------------
        # 1. Create Q, K and V
        # --------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # --------------------------------------------------
        # 2. Split embedding into multiple heads
        # --------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Shape:
        # [batch, heads, sequence, head_dim]

        # --------------------------------------------------
        # 3. Calculate attention scores
        # --------------------------------------------------

        scores = Q @ K.transpose(-2, -1)

        # Scale
        scores = scores / math.sqrt(self.head_dim)

        # --------------------------------------------------
        # 4. Apply causal mask
        # --------------------------------------------------

        causal_mask = self.mask[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        # --------------------------------------------------
        # 5. Convert scores to probabilities
        # --------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # --------------------------------------------------
        # 6. Weighted sum of values
        # --------------------------------------------------

        output = attention_weights @ V

        # --------------------------------------------------
        # 7. Put heads back together
        # --------------------------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # --------------------------------------------------
        # 8. Final projection
        # --------------------------------------------------

        output = self.output_projection(output)

        return output