import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax

def run_soft_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft KMeans 실행
    -------------------
    X          : (N, D) 임베딩 벡터
    n_clusters : 클러스터 개수
    n_init     : KMeans 초기화 반복 횟수
    random_state : 랜덤 시드

    return
    -------
    P            : (N, K) soft 확률
    C            : (K, D) 클러스터 중심
    y_pred_hard  : (N,) hard 클러스터 라벨
    """
    # KMeans 학습
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    y_pred_hard = kmeans.fit_predict(X)

    # 거리 계산 → soft 확률로 변환
    distances = kmeans.transform(X)  # (N, K)
    P = softmax(-distances, axis=1)  # 거리 음수 → softmax 확률

    C = kmeans.cluster_centers_

    return P, C, y_pred_hard
