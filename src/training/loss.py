import torch.nn as nn


class LanguageModelLoss:
    """
    Cross-entropy loss for next-token prediction.
    """

    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, logits, targets):
        """
        logits:
            [batch_size, sequence_length, vocab_size]

        targets:
            [batch_size, sequence_length]
        """

        batch_size, sequence_length, vocab_size = logits.shape

        # Flatten predictions
        logits = logits.view(
            batch_size * sequence_length,
            vocab_size
        )

        # Flatten target tokens
        targets = targets.view(
            batch_size * sequence_length
        )

        # Calculate loss
        loss = self.loss_function(
            logits,
            targets
        )

        return loss