from pathlib import Path
import numpy as np

from preprocess import preprocess, map_words
from functions import generate_negative_pair, generate_negative_prob, generate_positive_context
from model import SkipGramNS


def build_training_pairs(tokens, window_size):
    position_pairs = generate_positive_context(tokens, window_size)
    center_ids = np.asarray([tokens[center_pos] for center_pos, _ in position_pairs], dtype=np.int64)
    pos_context_ids = np.asarray([tokens[ctx_pos] for _, ctx_pos in position_pairs], dtype=np.int64)
    return center_ids, pos_context_ids


def train_skipgram_ns(center_ids, pos_context_ids, neg_probs, vocab_size, embed_dim=20, scale=0.1, learning_rate=0.03, epochs=5, K=5):
    model = SkipGramNS(V=vocab_size, d=embed_dim, scale=scale)
    num_samples = len(center_ids)

    for epoch in range(1, epochs + 1):
        neg_samples = generate_negative_pair(neg_probs, K=K, num_samples=num_samples)
        order = np.random.permutation(num_samples)

        for idx in order:
            model.backward_pass(
                center_id=center_ids[idx],
                pos_context_id=pos_context_ids[idx],
                neg_context_ids=neg_samples[idx],
                learning_rate=learning_rate,
            )

        epoch_loss = model.forward_pass(center_ids, pos_context_ids, neg_samples)
        print(f"epoch {epoch}/{epochs} - loss: {epoch_loss:.4f}")

    return model

def main():
    np.random.seed(42)

    path = Path("data/debugging.txt")
    text = path.read_text(encoding="utf-8")
    data = preprocess(text)
    word_to_id, id_to_word, tokens, frequency = map_words(data)

    window_size = 2
    center_ids, pos_context_ids = build_training_pairs(tokens, window_size)
    num_samples = len(center_ids)

    if num_samples == 0:
        raise ValueError("No positive pairs generated; check input text/window_size.")

    neg_probs = generate_negative_prob(frequency, 0.75)

    print(f"vocab_size: {len(word_to_id)}")
    print(f"tokens: {len(tokens)}")
    print(f"training_pairs: {num_samples}")

    train_skipgram_ns(
        center_ids=center_ids,
        pos_context_ids=pos_context_ids,
        neg_probs=neg_probs,
        vocab_size=len(word_to_id),
        embed_dim=20,
        scale=0.1,
        learning_rate=0.03,
        epochs=100,
        K=5,
    )


if __name__ == "__main__":
    main()
