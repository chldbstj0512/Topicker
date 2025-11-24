import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict

def hungarian_align(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    P: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Hungarian Algorithm으로 클러스터 라벨 정렬
    
    Parameters
    ----------
    y_pred : (N,) hard cluster labels (0 ~ K-1)
    y_true : (N,) ground truth labels
    P      : (N, K) soft probabilities
    C      : (K, D) cluster centers

    Returns
    -------
    y_aligned : (N,) aligned cluster labels
    mapping   : dict {pred_label -> true_label}
    """
    K = len(np.unique(y_true))
    cost_matrix = np.zeros((K, K), dtype=int)

    # cost_matrix[i, j] = # of samples where pred=i, true≠j
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = np.sum((y_pred == i) & (y_true != j))

    # Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}

    # 매핑 적용
    y_aligned = np.array([mapping[yi] for yi in y_pred], dtype=int)

    return y_aligned, mapping

def save_soft_results(
    path: str,
    P: np.ndarray,
    C: np.ndarray,
    y_pred_hard: np.ndarray,
    y_pred_aligned: np.ndarray,
    mapping: Dict[int, int],
):
    """
    soft_results 저장
    """
    np.savez(
        path,
        P=P,
        C=C,
        y_pred_hard=y_pred_hard,
        y_pred_aligned=y_pred_aligned,
        mapping=np.array(list(mapping.items()), dtype=int),
    )
    print(f"[Saved] {path}")
