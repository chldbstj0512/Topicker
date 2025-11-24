# relabel_pipeline.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Sequence, Optional
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

# 외부 의존 함수 (프로젝트 내 모듈)
from soft_kmeans import run_soft_kmeans
from hungarian_align import hungarian_align
# from get_embeddings import get_embeddings  # 필요 시 사용

# ---------------------------
# 0) 유틸
# ---------------------------
def _ensure_indices(indices: Sequence[int], n: int) -> np.ndarray:
    arr = np.asarray(indices, dtype=int)
    if not np.all((0 <= arr) & (arr < n)):
        raise ValueError("indices out of range.")
    return arr

# ---------------------------
# 1) 베이스라인 (Before)
# ---------------------------
def compute_baseline(
    X_all: np.ndarray,
    y_true: np.ndarray,
    K: int,
    target_idx: Sequence[int],
) -> Dict[str, object]:
    """
    기존 임베딩 X_all로 soft k-means → 헝가리안 정렬 → baseline 성능 산출
    반환: dict(P, C, y_hard, y_aligned, acc_targets)
    """
    target_idx = _ensure_indices(target_idx, len(X_all))
    P_bef, C_bef, y_pred_hard_bef = run_soft_kmeans(X_all, n_clusters=K)
    y_pred_aligned_bef, mapping_bef = hungarian_align(y_pred_hard_bef, y_true, P_bef, C_bef)

    acc_before_targets = accuracy_score(
        y_true[target_idx],
        y_pred_aligned_bef[target_idx]
    )

    return dict(
        P=P_bef, C=C_bef,
        y_hard=y_pred_hard_bef,
        y_aligned=y_pred_aligned_bef,
        mapping=mapping_bef,
        acc_targets=acc_before_targets,
    )

# ---------------------------
# 2) LLM 임베딩 준비
# ---------------------------
def prepare_llm_embeddings(
    df: pd.DataFrame,
    indices: Sequence[int],
    X_llm_block: Optional[np.ndarray] = None,
    *,
    use_label_plus_rationale: bool = False,
    text_label_col: str = "llm_label",
    text_rationale_col: str = "llm_rationale",
    # embed_fn: Optional[callable] = get_embeddings,  # 필요 시 활성화
) -> np.ndarray:
    """
    indices에 해당하는 행만 LLM 임베딩을 준비.
    - 이미 계산된 임베딩이 있다면 X_llm_block으로 전달(권장, 빠름)
    - 텍스트로부터 새로 임베딩하려면 embed_fn 사용 (주석 해제 필요)

    반환: (M, d) 의 LLM 임베딩 (M=len(indices))
    """
    idx = _ensure_indices(indices, len(df))

    if X_llm_block is not None:
        return np.asarray(X_llm_block, dtype=np.float32)

    # 텍스트 기반 임베딩이 필요한 경우에만 사용
    # if embed_fn is None:
    #     raise ValueError("Either X_llm_block or embed_fn must be provided.")
    # if use_label_plus_rationale:
    #     texts = (
    #         df.iloc[idx][text_label_col].fillna("").astype(str) + " " +
    #         df.iloc[idx][text_rationale_col].fillna("").astype(str)
    #     ).tolist()
    # else:
    #     texts = df.iloc[idx][text_label_col].fillna("").astype(str).tolist()
    # X_llm_block = embed_fn(texts)
    # return np.asarray(X_llm_block, dtype=np.float32)

    raise ValueError("X_llm_block을 전달하거나 텍스트 임베딩 경로를 활성화하세요.")

# ---------------------------
# 3) 임베딩 결합
# ---------------------------
def combine_embeddings(
    X_all_orig: np.ndarray,
    indices: Sequence[int],
    X_llm_block: np.ndarray,
    *,
    use_l2_normalized_mean: bool = True,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    대상 indices에 대해 원본 임베딩과 LLM 임베딩을 결합하여 X_all_updated 반환.
    - use_l2_normalized_mean=True: 행 단위 L2 정규화 후 가중 평균 → 다시 L2 정규화
    - False: 단순 가중합
    """
    X_all_orig = np.asarray(X_all_orig, dtype=np.float32)
    n, d = X_all_orig.shape
    idx = _ensure_indices(indices, n)

    X_orig_targets = X_all_orig[idx]
    X_llm_block = np.asarray(X_llm_block, dtype=np.float32)
    if X_llm_block.shape != X_orig_targets.shape:
        raise ValueError(f"shape mismatch: orig {X_orig_targets.shape} vs llm {X_llm_block.shape}")

    if use_l2_normalized_mean:
        X_a = normalize(X_orig_targets)   # (M, d)
        X_b = normalize(X_llm_block)
        X_comb = normalize((1.0 - alpha) * X_a + alpha * X_b)
    else:
        X_comb = (1.0 - alpha) * X_orig_targets + alpha * X_llm_block

    X_all_updated = X_all_orig.copy()
    X_all_updated[idx] = X_comb
    return X_all_updated

# ---------------------------
# 4) 클러스터링 & 정렬 (After)
# ---------------------------
def run_cluster_and_align(
    X_all: np.ndarray,
    y_true: np.ndarray,
    K: int,
) -> Dict[str, object]:
    """
    X_all로 soft k-means → 헝가리안 정렬
    반환: dict(P, C, y_hard, y_aligned, mapping)
    """
    P, C, y_pred_hard = run_soft_kmeans(X_all, n_clusters=K)
    y_pred_aligned, mapping = hungarian_align(y_pred_hard, y_true, P, C)
    return dict(P=P, C=C, y_hard=y_pred_hard, y_aligned=y_pred_aligned, mapping=mapping)

# ---------------------------
# 5) 대상 구간 정확도 비교
# ---------------------------
def evaluate_targets_acc(
    y_true: np.ndarray,
    y_before_aligned: np.ndarray,
    y_after_aligned: np.ndarray,
    indices: Sequence[int],
) -> Dict[str, float]:
    idx = _ensure_indices(indices, len(y_true))
    acc_before = accuracy_score(y_true[idx], y_before_aligned[idx])
    acc_after  = accuracy_score(y_true[idx], y_after_aligned[idx])
    return dict(before=acc_before, after=acc_after, delta=acc_after - acc_before)

# ---------------------------
# 6) NPZ 저장
# ---------------------------
def save_soft_results_npz(
    path: str,
    P: np.ndarray,
    C: np.ndarray,
    y_pred_hard: np.ndarray,
    y_pred_aligned: np.ndarray,
    mapping: Dict[int, int],
) -> None:
    """
    soft 결과를 통째로 저장 (load_soft_results() 호환)
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        P=P, C=C,
        y_pred_hard=y_pred_hard,
        y_pred_aligned=y_pred_aligned,
        mapping=np.array(list(mapping.items()), dtype=int),
    )
