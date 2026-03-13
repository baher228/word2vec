from preprocess import preprocess, map_words
from functions import generate_negative_pair, generate_negative_prob, generate_positive_context

def main():
    data = preprocess()
    word_to_id, id_to_word, tokens, frequency = map_words(data)
    batch_size = 32
    window_size = 2
    pos_pairs = generate_positive_context(tokens, window_size)
    neg_probs = generate_negative_prob(frequency, 0.75)
    neg_pairs = generate_negative_pair(neg_probs, K=5, batch_size=batch_size)