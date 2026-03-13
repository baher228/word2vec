from pathlib import Path
from preprocess import preprocess, map_words
from functions import generate_negative_pair, generate_negative_prob, generate_positive_context

def main():
    path = Path("data/debugging.txt")
    text = path.read_text(encoding="utf-8")
    data = preprocess(text)
    word_to_id, id_to_word, tokens, frequency = map_words(data)
    window_size = 2
    pos_pairs = generate_positive_context(tokens, window_size)
    num_samples = len(pos_pairs)
    neg_probs = generate_negative_prob(frequency, 0.75)
    neg_samples = generate_negative_pair(neg_probs, K=5, num_samples=num_samples)

    print(f'vocab_size: {len(word_to_id)}')
    print(f'tokens: {len(tokens)}')
    print(f'positive_pairs: {len(pos_pairs)}')
    print(f'negative_samples shape: {neg_samples.shape}')
    print('first 10 positive pairs:', pos_pairs[:10])
    print('first 3 negative rows:', neg_samples[:3])

if __name__ == "__main__":
    main()

