import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer


def get_word_embeddings(feature_data, context_length=50, model_name="gpt2"):
    """
    Generate contextual word embeddings using GPT-2 hidden states.

    Parameters
    ----------
    feature_data : dict
        Mapping from onset -> feature dict containing 'speech'
    context_length : int
        Number of previous words used as context
    model_name : str
        HuggingFace model name

    Returns
    -------
    dict
        onset -> embedding vector
    """

    print("Loading GPT-2 model and tokenizer...")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    model.eval()

    # Convert feature_data to ordered list
    ordered_words = [(onset, feat['speech']) for onset, feat in feature_data.items()]
    total_words = len(ordered_words)

    print(f"Generating embeddings for {total_words} words")

    word_embeddings = {}
    skipped = 0

    for i, (onset, word) in enumerate(ordered_words):

        try:
            start_idx = max(0, i - context_length)

            context_words = [
                ordered_words[j][1] for j in range(start_idx, i)
            ]
            context_words.append(word)

            input_text = " ".join(context_words)

            tokens = tokenizer.encode(input_text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(tokens, output_hidden_states=True)
                embedding = outputs.hidden_states[-1][:, -1, :].squeeze().numpy()

            word_embeddings[onset] = embedding

            if (i + 1) % 500 == 0:
                print(f"{i+1}/{total_words} words processed")

        except Exception as e:
            print(f"Error at word '{word}': {e}")
            skipped += 1

    print(f"Done. Embeddings: {len(word_embeddings)} | Skipped: {skipped}")

    return word_embeddings