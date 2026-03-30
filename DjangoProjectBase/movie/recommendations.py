from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from django.conf import settings

from .models import Movie


_CLIENT: Optional[OpenAI] = None
_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    candidates: list[Path] = []

    try:
        base_dir = Path(settings.BASE_DIR)
        candidates.append(base_dir.parent / "openAI.env")
        candidates.append(base_dir / "openAI.env")
    except Exception:
        pass

    candidates.append(Path("openAI.env"))

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path)
            _ENV_LOADED = True
            return

    load_dotenv()
    _ENV_LOADED = True


def get_openai_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    _load_env_once()

    api_key = (
        os.environ.get("openai_apikey")
        or os.environ.get("openai_api_key")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Add 'openai_apikey' (or 'openai_api_key') to openAI.env."
        )

    _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    text = (text or "").replace("\n", " ")
    client = get_openai_client()
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)


def recommend_movie_for_prompt(prompt: str) -> Tuple[Optional[Movie], Optional[float]]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None, None

    prompt_emb = get_embedding(prompt)
    dims = int(prompt_emb.shape[0])

    best_movie: Optional[Movie] = None
    best_similarity = -1.0

    for movie in Movie.objects.all().only(
        "id",
        "title",
        "description",
        "image",
        "url",
        "genre",
        "year",
        "emb",
    ).iterator():
        raw = bytes(movie.emb) if movie.emb is not None else b""
        if not raw or len(raw) % 4 != 0:
            continue

        movie_emb = np.frombuffer(raw, dtype=np.float32)
        if int(movie_emb.shape[0]) != dims:
            continue

        similarity = cosine_similarity(prompt_emb, movie_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_movie = movie

    if best_movie is None:
        return None, None

    return best_movie, best_similarity
