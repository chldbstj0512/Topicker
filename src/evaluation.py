# evaluation.py
import os
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# -------------------------
# External metrics
# -------------------------
def purity_p1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    P@1 (Purity): 각 예측 클러스터에서 최빈 진짜 라벨 개수를 합산 / 전체 샘플 수
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape[0] == y_pred.shape[0]
    N = y_true.shape[0]
    total = 0
    for k in np.unique(y_pred):
        mask = (y_pred == k)
        if mask.sum() == 0:
            continue
        majority = Counter(y_true[mask]).most_common(1)[0][1]
        total += majority
    return total / N

def ext_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "P1": purity_p1(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred)
    }

# -------------------------
# Internal metrics
# -------------------------
def topic_diversity(topics_words: List[List[str]], topn: int = 10) -> Optional[float]:
    """
    TD: 고유 단어 수 / (K * topn)
    """
    if not topics_words:
        return None
    trimmed = [tw[:topn] for tw in topics_words if tw]
    if not trimmed:
        return None
    K = len(trimmed)
    all_words = [w for tw in trimmed for w in tw]
    if K * topn == 0:
        return None
    return len(set(all_words)) / (K * topn)

# -------------------------
# Tokenization helpers (for c-TF-IDF)
# -------------------------
def _default_tokenize(text: str, min_word_len: int = 2) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return [t for t in tokens if len(t) >= min_word_len]

def _generate_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
    nmin, nmax = ngram_range
    out = []
    for n in range(nmin, nmax + 1):
        if n == 1:
            out.extend(tokens)
        else:
            for i in range(len(tokens) - n + 1):
                out.append("_".join(tokens[i:i+n]))
    return out

# -------------------------
# Build topics with c-TF-IDF
# -------------------------
def build_topics_from_clusters_ctfidf(
    texts: Union[List[str], List[List[str]]],
    labels_pred: List[int],
    topn: int = 10,
    min_word_len: int = 4,
    ngram_range: Tuple[int, int] = (1, 1),
    include_noise: bool = False,
    stop_words: Optional[Union[set, List[str]]] = None,
    hangul_mode: bool = False,
    return_scores: bool = False
) -> List[List[str]]:
    assert len(texts) == len(labels_pred), "texts와 labels_pred 길이가 다릅니다."

    stop_set = set(stop_words) if stop_words is not None else None

    # 클러스터별 문서 모으기
    cluster_docs: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels_pred):
        if lab == -1 and not include_noise:
            continue
        cluster_docs[lab].append(idx)

    if not cluster_docs:
        return []

    is_pretok = isinstance(texts[0], (list, tuple))
    tokenizer = _default_tokenize

    # 클러스터별 토큰 수집
    cluster_tokens: Dict[int, List[str]] = {}
    for cid, doc_indices in cluster_docs.items():
        toks: List[str] = []
        for di in doc_indices:
            ts = texts[di] if is_pretok else tokenizer(texts[di], min_word_len)
            if stop_set:
                ts = [w for w in ts if w not in stop_set]
            toks.extend(_generate_ngrams(ts, ngram_range))
        cluster_tokens[cid] = toks

    # vocab 구축
    vocab = {}
    for toks in cluster_tokens.values():
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)
    inv_vocab = {j: w for w, j in vocab.items()}

    V = len(vocab)
    if V == 0:
        return [[] for _ in cluster_tokens]

    # TF
    counts = {cid: [0] * V for cid in cluster_tokens.keys()}
    total_words_per_class = {}
    for cid, toks in cluster_tokens.items():
        row = counts[cid]
        for w in toks:
            row[vocab[w]] += 1
        total_words_per_class[cid] = sum(row)

    # c-TF-IDF
    tf_t = [0] * V
    for row in counts.values():
        for j in range(V):
            tf_t[j] += row[j]

    C = len(cluster_tokens)
    A = sum(total_words_per_class.values()) / float(C)
    inv_class_part = [math.log(1.0 + (A / tf_t[j])) if tf_t[j] > 0 else 0.0 for j in range(V)]

    results = []
    for cid in sorted(cluster_tokens.keys()):
        row = counts[cid]
        scores = [(j, row[j] * inv_class_part[j]) for j in range(V) if row[j] > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = scores[:topn]
        if return_scores:
            results.append([(inv_vocab[j], float(s)) for j, s in top_items])
        else:
            results.append([inv_vocab[j] for j, _ in top_items])
    return results

# -------------------------
# Wrapper: evaluate multiple models
# -------------------------
def evaluate_models(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    texts: Optional[List[str]] = None,
    topn: int = 10,
    save_topic_path: Optional[str] = None
) -> Dict[str, Dict[str, Optional[float]]]:
    results = {}
    topics_words_all = {}

    for name, y_pred in preds.items():
        # External
        e = ext_metrics(y_true, y_pred)

        # Internal (topic words + TD)
        td = None
        if texts is not None:
            topics_words = build_topics_from_clusters_ctfidf(texts, y_pred, topn=topn)
            topics_words_all[name] = topics_words
            td = topic_diversity(topics_words, topn=topn)

            # Save topic words if path given
            if save_topic_path:
                with open(save_topic_path, "w", encoding="utf-8") as f:
                    for topic_id, words in enumerate(topics_words):
                        f.write(f"Topic {topic_id}: {', '.join(words)}\n")

        results[name] = {
            "P1": e["P1"],
            "ARI": e["ARI"],
            "NMI": e["NMI"],
            "TD": td
        }
    return results
