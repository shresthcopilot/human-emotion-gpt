import torch


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    context_length,
    temperature=1.0,
    top_k=None,
    device="cpu"
):
    """
    Generate text from the MiniGPT model.

    Parameters:
        temperature:
            Controls randomness.

            < 1.0 = more predictable
            1.0   = normal
            > 1.0 = more random

        top_k:
            Only consider the top K most likely tokens.

            None = consider all tokens
    """

    model.eval()

    # Convert prompt into token IDs
    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            input_context = input_ids[
                :, -context_length:
            ]

            # Model prediction
            logits = model(input_context)

            # We only need predictions for the last token
            next_token_logits = logits[:, -1, :]

            # ------------------------------------------
            # Temperature
            # ------------------------------------------

            if temperature <= 0:
                raise ValueError(
                    "Temperature must be greater than 0."
                )

            next_token_logits = (
                next_token_logits / temperature
            )

            # ------------------------------------------
            # Top-K filtering
            # ------------------------------------------

            if top_k is not None:

                top_k = min(
                    top_k,
                    next_token_logits.size(-1)
                )

                values, _ = torch.topk(
                    next_token_logits,
                    top_k
                )

                minimum_value = values[:, -1].unsqueeze(-1)

                next_token_logits = torch.where(
                    next_token_logits < minimum_value,
                    torch.full_like(
                        next_token_logits,
                        float("-inf")
                    ),
                    next_token_logits
                )

            # ------------------------------------------
            # Convert logits into probabilities
            # ------------------------------------------

            probabilities = torch.softmax(
                next_token_logits,
                dim=-1
            )

            # ------------------------------------------
            # Sample next token
            # ------------------------------------------

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Add new token to sequence
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

    generated_ids = input_ids[0].tolist()

    return tokenizer.decode(generated_ids)