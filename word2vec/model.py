import numpy as np
from functions import sigmoid

class SkipGramNS:
    def __init__(self, V, d, scale):
        self.W_in  = np.random.normal(loc=0.0, scale=scale, size=(V, d))
        self.W_out = np.random.normal(loc=0.0, scale=scale, size=(V, d))
    def forward_pass(center_ids, pos_context_ids, neg_context_ids, id2word, K):
        sc_pos = 0
        sc_neg = 0
        for i in range(len(center_ids)):
            v_center = id2word[center_ids[i]]
            v_pos = id2word[pos_context_ids[i]]
            v_neg = id2word[neg_context_ids[i]]
            sc_pos += np.log(sigmoid(np.dot(v_center, v_pos)))
            for j in range(K):
                sc_neg += np.log(sigmoid(np.dot(v_center, v_neg[j])))
