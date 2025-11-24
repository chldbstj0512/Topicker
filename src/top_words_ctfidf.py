# top_words_ctfidf.py
import math
import re
from collections import defaultdict
from typing import Iterable, List, Tuple, Dict, Optional, Union

def _default_tokenize(text: str, min_word_len: int = 2) -> List[str]:
    """영문 기본 토크나이저. (한글 포함하려면 hangul_mode=True 옵션 사용)"""
    tokens = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return [t for t in tokens if len(t) >= min_word_len]

def _generate_ngrams(tokens: List[str], ngram_range: Tuple[int, int]) -> Iterable[str]:
    """토큰 시퀀스에서 ngram 생성 (공백 대신 '_'로 join)."""
    nmin, nmax = ngram_range
    for n in range(nmin, nmax + 1):
        if n == 1:
            for tok in tokens:
                yield tok
        else:
            L = len(tokens)
            if L < n:
                continue
            for i in range(L - n + 1):
                yield "_".join(tokens[i:i + n])

def build_topics_from_clusters_ctfidf(
    texts: Union[List[str], List[List[str]]],     # 문자열 문서 or 사전 토큰화 문서
    labels_pred: List[int],                      # 각 문서의 클러스터 라벨
    topn: int = 10,
    min_word_len: int = 4,
    ngram_range: Tuple[int, int] = (1, 1),
    include_noise: bool = False,                 # 라벨 -1(노이즈) 포함 여부
    tokenizer: Optional[callable] = None,        # 문자열 문서일 때만 사용; 미지정 시 기본 토크나이저
    stop_words: Optional[Iterable[str]] = None,
    return_scores: bool = False,                 # True면 (단어, 점수) 반환
    apply_length_filter_on_pretokenized: bool = False,
    lowercase_pretokenized: bool = False,
    hangul_mode: bool = False                    # True면 영문+한글 토큰화
) -> List[List[str]]:
    """
    c-TF-IDF 방식으로 각 클러스터(토픽)의 상위 단어 추출.

    Returns
    -------
    - return_scores=False: List[List[str]]
        예: [[토픽0-단어들...], [토픽1-단어들...], ...]
    - return_scores=True: List[List[Tuple[str, float]]]
        예: [[(단어, 점수), ...], ...]
    """
    assert len(texts) == len(labels_pred), "texts와 labels_pred 길이가 다릅니다."

    # 0) stopwords 준비
    stop_set = set(stop_words) if stop_words is not None else None

    # 1) 클러스터별 문서 인덱스 수집
    cluster_docs: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels_pred):
        if lab == -1 and not include_noise:
            continue
        cluster_docs[lab].append(idx)
    if not cluster_docs:
        return []

    # 2) 입력 형태 판별 (문자열 vs 사전 토큰화)
    is_pretokenized = isinstance(texts[0], (list, tuple))

    # 3) 클러스터별 토큰 모으기 (+ ngram 전개)
    cluster_tokens: Dict[int, List[str]] = {}
    for cid, doc_indices in cluster_docs.items():
        toks: List[str] = []
        for di in doc_indices:
            if is_pretokenized:
                ts = list(texts[di])  # shallow copy
                if lowercase_pretokenized:
                    ts = [t.lower() for t in ts]
                if apply_length_filter_on_pretokenized and min_word_len > 1:
                    ts = [t for t in ts if len(t) >= min_word_len]
            else:
                ts = tokenizer(texts[di] or "", min_word_len=min_word_len)

            # stopwords 제거
            if stop_set:
                ts = [w for w in ts if w not in stop_set]

            # n-gram으로 확장
            for tok in _generate_ngrams(ts, ngram_range):
                if (not stop_set) or (tok not in stop_set):
                    toks.append(tok)

        cluster_tokens[cid] = toks

    # 4) 어휘 사전 구축
    vocab: Dict[str, int] = {}
    for toks in cluster_tokens.values():
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)
    C = len(cluster_tokens)
    if V == 0:
        # 토큰이 전혀 없으면 클러스터 수만큼 빈 리스트 반환
        return [[] for _ in range(C)]

    # 5) 클러스터-단어 카운트 (TF)
    counts = {cid: [0] * V for cid in cluster_tokens.keys()}
    total_words_per_class = {}
    for cid, toks in cluster_tokens.items():
        row = counts[cid]
        for w in toks:
            row[vocab[w]] += 1
        total_words_per_class[cid] = sum(row)

    # 6) c-TF-IDF 계수 계산
    #   문헌에 따라 class-based TF-IDF = (term count in class) * log(1 + A / tf_t)
    #   A = 전체 토큰 수 / 클러스터 수, tf_t = 전체 클러스터에서의 term 총 빈도
    tf_t = [0] * V
    for row in counts.values():
        for j in range(V):
            tf_t[j] += row[j]

    A = (sum(total_words_per_class.values()) / float(C)) if C > 0 else 0.0
    inv_class_part = [0.0] * V  # log(1 + A / tf_t)
    for j in range(V):
        inv_class_part[j] = math.log(1.0 + (A / tf_t[j])) if tf_t[j] > 0 else 0.0

    inv_vocab = {idx: w for w, idx in vocab.items()}

    # 7) 각 클러스터별 상위 단어 추출
    #    cid 오름차순으로 결과를 정렬해 일관성 유지
    results: List[List[str]] = []
    for cid in sorted(cluster_tokens.keys()):
        row = counts[cid]
        # 점수 = class term count * inv_class_part
        scores = [(j, row[j] * inv_class_part[j]) for j in range(V) if row[j] > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_items = scores[:topn]
        if return_scores:
            results.append([(inv_vocab[j], float(s)) for j, s in top_items])
        else:
            results.append([inv_vocab[j] for j, _ in top_items])

    return results
