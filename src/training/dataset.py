import torch


class RandomTextBatcher:
    """
    Efficient random batch sampler for language-model training.

    Instead of creating millions of overlapping samples,
    we randomly select starting positions from the token sequence.

    Example:

        tokens: A B C D E F G H

        context_length = 4

        possible samples:
            A B C D -> B C D E
            B C D E -> C D E F
            C D E F -> D E F G
            ...
    """

    def __init__(
        self,
        token_ids,
        context_length,
        batch_size,
        device="cpu"
    ):
        self.data = torch.tensor(token_ids, dtype=torch.long)

        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def get_batch(self):
        """
        Return one random training batch.

        x = input tokens
        y = next-token targets
        """

        max_start = len(self.data) - self.context_length - 1

        starts = torch.randint(
            0,
            max_start,
            (self.batch_size,)
        )

        offsets = torch.arange(self.context_length)

        x = self.data[
            starts[:, None] + offsets
        ]

        y = self.data[
            starts[:, None] + offsets + 1
        ]

        return (
            x.to(self.device),
            y.to(self.device)
        )