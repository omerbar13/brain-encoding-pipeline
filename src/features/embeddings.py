# src/features/embeddings.py

import logging
import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

logger = logging.getLogger(__name__)


def get_word_embeddings(
    feature_data: dict,
    context_length: int = 50,
    model_name: str = "gpt2",
) -> dict:
    """
    Generate contextual word embeddings using GPT-2 hidden states.

    For each word, the last num_tokens hidden state of the final layer
    is used as the embedding — this captures the word in context of the
    preceding context_length words.

    Tokenisation matches the notebook exactly: each word is prefixed with
    a leading space before joining (' ' + word), which aligns with GPT-2's
    byte-pair encoding convention and produces the correct subword tokens.

    Parameters
    ----------
    feature_data : dict
        onset (float) -> {'speech': str, 'duration': float, ...}
        Output of process_annotations().
    context_length : int
        Number of preceding words used as context (default 50).
    model_name : str
        HuggingFace model identifier (default 'gpt2').

    Returns
    -------
    dict
        onset (float) -> np.ndarray of shape (768,)
    """
    logger.info("Loading GPT-2 tokenizer and model: %s", model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model     = GPT2Model.from_pretrained(model_name)
    model.eval()

    ordered_words = [(onset, feat['speech']) for onset, feat in feature_data.items()]
    total_words   = len(ordered_words)
    logger.info("Generating embeddings for %d words", total_words)

    word_embeddings = {}
    skipped         = 0

    for i, (onset, word) in enumerate(ordered_words):
        try:
            start_idx = max(0, i - context_length)

            # Leading-space prefix on every word — matches GPT-2's BPE convention.
            # e.g. "hello world" is tokenised as ["hello", "Ġworld"] not ["hello", "world"]
            # This matches the notebook's: [' ' + w for w in context] + [' ' + word]
            context_words = [' ' + w for _, w in ordered_words[start_idx:i]]
            context_words.append(' ' + word)
            input_seq = ''.join(context_words)

            tokens = tokenizer.encode(input_seq, return_tensors='pt')

            with torch.no_grad():
                outputs   = model(tokens, output_hidden_states=True)
                # [:, -1, :] — last token position = current word in context
                embedding = outputs.hidden_states[-1][:, -1, :].squeeze().numpy()

            word_embeddings[onset] = embedding

            if (i + 1) % 500 == 0:
                logger.info(
                    "Embeddings progress: %d/%d (%.1f%%)",
                    i + 1, total_words, (i + 1) / total_words * 100
                )

        except Exception as e:
            logger.warning("Error at word '%s' (onset %.2f): %s", word, onset, e)
            skipped += 1

    logger.info(
        "Embeddings complete | generated: %d | skipped: %d",
        len(word_embeddings), skipped
    )

    if len(word_embeddings) != total_words:
        missing = total_words - len(word_embeddings)
        logger.warning(
            "%d words (%.1f%%) are missing embeddings",
            missing, missing / total_words * 100
        )

    return word_embeddings