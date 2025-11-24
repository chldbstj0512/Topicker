# get_embeddings.py
import os
from typing import List, Dict, Union

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

# --- 환경 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 모델 및 토큰 설정 ---
MODEL_NAME = "text-embedding-3-small"
MAX_TOKENS = 8192
enc = tiktoken.encoding_for_model(MODEL_NAME)

def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """텍스트를 모델 최대 토큰 길이에 맞게 자르기"""
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

def get_embed_dim(model_name: str) -> int:
    """모델 이름에 따른 임베딩 차원 반환"""
    name = model_name.lower()
    if "text-embedding-3-large" in name:
        return 3072
    return 1536  # small / ada 계열 기본값

def get_embeddings(
    texts: List[Union[str, List[str]]],
    model: str = MODEL_NAME,
    verbose: bool = True,
) -> np.ndarray:
    """
    텍스트 리스트 → 임베딩 벡터 배열 반환
    실패한 항목은 0-벡터로 채움
    """
    N = len(texts)
    D = get_embed_dim(model)
    X = np.zeros((N, D), dtype=np.float32)

    ok_idx, empty_idx, fail_idx = [], [], []

    for idx, raw in enumerate(tqdm(texts, desc="Embedding", total=N)):
        # 리스트면 문자열로 합치기
        text = " ".join(map(str, raw)) if isinstance(raw, list) else str(raw)
        text = truncate_text(text.strip())

        if not text:
            empty_idx.append(idx)
            continue

        try:
            resp = client.embeddings.create(model=model, input=text)
            vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
            X[idx] = vec
            ok_idx.append(idx)
        except Exception as e:
            if verbose:
                print(f"[warn] idx={idx} embedding failed: {e!r}")
            fail_idx.append(idx)

    if verbose:
        print("\n=== Embedding Summary ===")
        print(f"Total inputs   : {N}")
        print(f"OK             : {len(ok_idx)}")
        print(f"Empty texts    : {len(empty_idx)}")
        print(f"API failures   : {len(fail_idx)}")
        print(f"Kept (non-zero): {(X.sum(axis=1)!=0).sum()}")

    return X
