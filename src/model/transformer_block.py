import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single Transformer block.

    Structure:

        LayerNorm
            ↓
        Multi-Head Attention
            ↓
        Residual Connection
            ↓
        LayerNorm
            ↓
        Feed-Forward Network
            ↓
        Residual Connection
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        context_length,
        hidden_dim
    ):
        super().__init__()

        # Layer normalization before attention
        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        # Multi-head causal self-attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length
        )

        # Layer normalization before feed-forward network
        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x):

        # --------------------------------
        # Attention + residual connection
        # --------------------------------

        normalized_x = self.layer_norm_1(x)

        attention_output = self.attention(
            normalized_x
        )

        x = x + attention_output

        # --------------------------------
        # Feed-forward + residual connection
        # --------------------------------

        normalized_x = self.layer_norm_2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        x = x + feed_forward_output

        return x