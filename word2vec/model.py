import numpy as np
from functions import sigmoid

class SkipGramNS:
    def __init__(self, V, d, scale):
        self.W_in  = np.random.normal(loc=0.0, scale=scale, size=(V, d))
        self.W_out = np.random.normal(loc=0.0, scale=scale, size=(V, d))

    def _scores_and_probs(self, center_id, pos_context_id, neg_context_ids):
        center_id = int(center_id)
        pos_context_id = int(pos_context_id)
        neg_context_ids = np.asarray(neg_context_ids, dtype=np.int64)

        if neg_context_ids.ndim != 1:
            raise ValueError("neg_context_ids must be a 1D array-like of negative ids")

        u = self.W_in[center_id]               # (d,)
        v_pos = self.W_out[pos_context_id]     # (d,)
        v_negs = self.W_out[neg_context_ids]   # (K, d)

        s_pos = np.dot(u, v_pos)
        s_negs = np.dot(v_negs, u)

        y_pos = sigmoid(s_pos)
        y_negs = sigmoid(s_negs)

        return u, v_pos, v_negs, s_pos, s_negs, y_pos, y_negs, neg_context_ids

    def forward_pass(self, center_ids, pos_context_ids, neg_context_ids):
        sc_pos = 0
        sc_neg = 0

        for i in range(len(center_ids)):
            _, _, _, s_pos, s_negs, _, _, _ = self._scores_and_probs(
                center_id=center_ids[i],
                pos_context_id=pos_context_ids[i],
                neg_context_ids=neg_context_ids[i],
            )
            sc_pos += np.log(sigmoid(s_pos))
            sc_neg += np.sum(np.log(sigmoid(-s_negs)))

        loss = -(sc_pos + sc_neg) / len(center_ids)
        return loss

    def backward_pass(self, center_id, pos_context_id, neg_context_ids, learning_rate):
        u, v_pos, v_negs, _, _, y_pos, y_negs, neg_context_ids = self._scores_and_probs(
            center_id=center_id,
            pos_context_id=pos_context_id,
            neg_context_ids=neg_context_ids,
        )

        grad_u = (y_pos - 1.0) * v_pos + np.sum(y_negs[:, None] * v_negs, axis=0)
        grad_v_pos = (y_pos - 1.0) * u
        grad_v_negs = y_negs[:, None] * u

        self.W_in[center_id] -= learning_rate * grad_u
        self.W_out[pos_context_id] -= learning_rate * grad_v_pos
        np.add.at(self.W_out, neg_context_ids, -learning_rate * grad_v_negs)

        return grad_u, grad_v_pos, grad_v_negs
