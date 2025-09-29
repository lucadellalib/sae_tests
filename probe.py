import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

# utterances[i] = list of frames; each frame is (idxs_t, vals_t)
# idxs_t: np.int32[k] in [0, V-1], vals_t: np.float32[k]
V = 62000

def features_counts(utt, V, length_norm=True, binary=False):
    """Return 1xV CSR with counts (or binary presence)."""
    T = len(utt)
    if T == 0: return csr_matrix((1, V), dtype=np.float32)

    acc = {}
    for idxs_t, vals_t in utt:
        if binary:
            for j in idxs_t:
                acc[j] = 1.0
        else:
            for j in idxs_t:
                acc[j] = acc.get(j, 0.0) + 1.0

    if not acc: return csr_matrix((1, V), dtype=np.float32)
    idxs = np.fromiter(acc.keys(), dtype=np.int32)
    data = np.fromiter(acc.values(), dtype=np.float32)
    if length_norm and not binary:
        data /= float(T)  # rate per frame
    indptr = np.array([0, len(idxs)], dtype=np.int32)
    return csr_matrix((data, idxs, indptr), shape=(1, V), dtype=np.float32)

def features_strength(utt, V, length_norm=True):
    """Return 1xV CSR with sum of activation magnitudes."""
    T = len(utt)
    if T == 0: return csr_matrix((1, V), dtype=np.float32)

    acc = {}
    for idxs_t, vals_t in utt:
        for j, v in zip(idxs_t, vals_t):
            acc[j] = acc.get(j, 0.0) + float(v)

    if not acc: return csr_matrix((1, V), dtype=np.float32)
    idxs = np.fromiter(acc.keys(), dtype=np.int32)
    data = np.fromiter(acc.values(), dtype=np.float32)
    if length_norm:
        data /= float(T)
    indptr = np.array([0, len(idxs)], dtype=np.int32)
    return csr_matrix((data, idxs, indptr), shape=(1, V), dtype=np.float32)

# After you’ve built X (N x V CSR), compute IDF and apply:
def apply_idf(X):
    N, V = X.shape
    # df: number of nonzeros per column
    df = (X != 0).astype(np.int8).sum(axis=0).A1
    idf = np.log((N + 1) / (df + 1)).astype(np.float32)
    # scale each column
    X = X.tocsc(copy=True)
    X.data *= np.repeat(idf, np.diff(X.indptr))
    return X.tocsr()


def train_eval(X, y):
    clf = LogisticRegression(
        penalty="l1", solver="saga",
        C=0.5, max_iter=5000, n_jobs=-1,
        class_weight="balanced",
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    aps = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], p))
    print(f"PR-AUC mean±sd: {np.mean(aps):.3f} ± {np.std(aps):.3f}")
    return clf
