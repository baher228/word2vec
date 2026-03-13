import numpy as np

def generate_positive_context(tokens, window_size):
    positive_pairs = []
    for i in range(len(tokens)):
        for j in range(i - window_size, i + window_size + 1):
            if j >= 0 and j < len(tokens) and j != i:
                positive_pairs.append((i, j))
    return positive_pairs
    
def generate_negative_prob(frequency, power):
    freq = np.asarray(frequency, np.float64)
    probs = freq ** power
    Z = probs.sum()
    return probs / Z
    
def generate_negative_pair(probs, K, num_samples):
    V = probs.shape[0]
    return np.random.choice(V, size=(num_samples, K), replace=True, p=probs)

def sigmoid(val):
    return 1 / (1 + np.exp(-val))
