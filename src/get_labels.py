# get_labels.py
import os
import json
import time
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ------------------------
# LLM Client
# ------------------------
def _get_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAI(api_key=key)


# ------------------------
# Token utils
# ------------------------
def truncate_by_tokens(text: str, max_tokens: int = 3000) -> str:
    if tiktoken is None:
        return (text or "")[: max_tokens * 4]
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text or "")
    if len(ids) <= max_tokens:
        return text or ""
    return enc.decode(ids[:max_tokens])


# ------------------------
# Prompt
# ------------------------
def build_prompt(text: str) -> str:
    return (
        "You are tasked with inferring a latent topic representation from a given text.\n"
        "Analyze the semantic content holistically and propose a concise latent topic label.\n\n"
        "INPUT:\n"
        f'- text: "{text}"\n\n'
        "TASK:\n"
        "1. Consider a short latent topic label that captures the main semantic theme.\n"
        "2. Reflect internally on a brief rationale (1–2 sentences).\n"
        "3. Identify 5–8 salient keywords.\n"
        "4. Estimate a confidence score between 0 and 1.\n\n"
        "RESPONSE:\n(label only)"
    )



# ------------------------
# LLM Call
# ------------------------
def _call_llm_label(
    text: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 2,
    api_key: Optional[str] = None,
) -> str:
    client = _get_client(api_key)
    prompt = build_prompt(text)
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            label = resp.choices[0].message.content.strip()
            return label
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.0 * (2 ** attempt))
    print(f"[warn] LLM label call failed: {last_err!r}")
    return ""


# ------------------------
# Public Function
# ------------------------
def label_texts_only(
    texts: List[str],
    *,
    model: str = "gpt-4o",
    max_input_tokens: int = 3000,
    temperature: float = 0.0,
    max_retries: int = 2,
    api_key: Optional[str] = None,
    show_progress: bool = True,
) -> List[str]:
    """
    여러 텍스트에 대해 LLM이 생성한 label 값만 반환.
    """
    iterator = texts
    if show_progress and tqdm is not None:
        iterator = tqdm(texts, desc="Labeling", total=len(texts))

    results: List[str] = []
    for t in iterator:
        txt = truncate_by_tokens(t or "", max_input_tokens)
        label = _call_llm_label(
            txt,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            api_key=api_key,
        )
        results.append(label)
    return results
