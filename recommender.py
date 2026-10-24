import os
import pickle
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def load_model():
    """Load the movie DataFrame and its cosine-similarity matrix."""
    movies_path = MODEL_DIR / "movie_list.pkl"
    similarity_path = MODEL_DIR / "similarity.pkl"

    if not movies_path.exists() or not similarity_path.exists():
        missing = [str(p) for p in (movies_path, similarity_path) if not p.exists()]
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

    # Only load pickle files that you created or otherwise trust.
    with movies_path.open("rb") as file:
        movies = pickle.load(file)
    with similarity_path.open("rb") as file:
        similarity = pickle.load(file)

    if "title" not in movies.columns or "movie_id" not in movies.columns:
        raise ValueError("movie_list.pkl must contain 'title' and 'movie_id' columns.")
    if len(movies) != len(similarity):
        raise ValueError(
            f"Model mismatch: {len(movies)} movies but similarity has {len(similarity)} rows. "
            "Regenerate both files using build_model.py."
        )
    return movies.reset_index(drop=True), similarity


def recommend_movies(title, movies, similarity, n=5):
    """Return up to n similar movies, excluding the selected movie itself."""
    matches = movies.index[movies["title"].astype(str) == str(title)].tolist()
    if not matches:
        raise ValueError(f"Movie not found: {title}")

    movie_index = matches[0]
    scores = list(enumerate(similarity[movie_index]))
    scores.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for index, score in scores:
        if index == movie_index:
            continue
        row = movies.iloc[index]
        results.append({
            "title": str(row["title"]),
            "movie_id": int(row["movie_id"]),
            "score": float(score),
        })
        if len(results) == n:
            break
    return results


def _get_tmdb_api_key():
    """Read the key from Streamlit secrets or an environment variable."""
    try:
        import streamlit as st
        key = st.secrets.get("TMDB_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY", "")


def fetch_poster(movie_id):
    """Return a TMDB poster URL, or None if no key/poster is available."""
    api_key = _get_tmdb_api_key()
    if not api_key:
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key},
            timeout=8,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except (requests.RequestException, ValueError):
        return None
