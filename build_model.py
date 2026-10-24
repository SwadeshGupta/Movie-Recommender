"""Build the content-based recommender from the TMDB 5000 CSV files."""
import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


def parse_names(value, limit=None):
    """Convert TMDB JSON-like string columns into a list of names."""
    if pd.isna(value):
        return []
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    names = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]).replace(" ", ""))
        if limit and len(names) >= limit:
            break
    return names


def get_director(value):
    if pd.isna(value):
        return []
    try:
        items = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return []
    if not isinstance(items, list):
        return []
    return [
        str(item["name"]).replace(" ", "")
        for item in items
        if isinstance(item, dict)
        and item.get("job") == "Director"
        and item.get("name")
    ][:1]


def main():
    movies_file = DATA_DIR / "tmdb_5000_movies.csv"
    credits_file = DATA_DIR / "tmdb_5000_credits.csv"

    if not movies_file.exists() or not credits_file.exists():
        raise FileNotFoundError(
            "Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in the data/ folder. "
            "See README.md for dataset instructions."
        )

    movies_raw = pd.read_csv(movies_file)
    credits_raw = pd.read_csv(credits_file)

    # The credits file commonly calls its movie identifier 'movie_id' in some copies,
    # while the standard TMDB 5000 credits file uses 'movie_id' and 'title'.
    if "movie_id" not in credits_raw.columns and "id" in credits_raw.columns:
        credits_raw = credits_raw.rename(columns={"id": "movie_id"})

    required_movies = {"id", "title", "overview", "genres", "keywords"}
    required_credits = {"movie_id", "title", "cast", "crew"}
    if not required_movies.issubset(movies_raw.columns):
        raise ValueError(f"Movies CSV is missing columns: {required_movies - set(movies_raw.columns)}")
    if not required_credits.issubset(credits_raw.columns):
        raise ValueError(f"Credits CSV is missing columns: {required_credits - set(credits_raw.columns)}")

    movies = movies_raw.merge(credits_raw, on="title", suffixes=("", "_credits"))
    movies = movies[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
    movies = movies.rename(columns={"id": "movie_id"})
    movies = movies.dropna(subset=["title"]).drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    movies["overview"] = movies["overview"].fillna("").astype(str).apply(lambda value: value.split())
    movies["genres"] = movies["genres"].apply(parse_names)
    movies["keywords"] = movies["keywords"].apply(parse_names)
    movies["cast"] = movies["cast"].apply(lambda value: parse_names(value, limit=3))
    movies["crew"] = movies["crew"].apply(get_director)

    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    ).apply(lambda parts: " ".join(str(part) for part in parts).lower())

    model_movies = movies[["movie_id", "title", "tags"]].copy().reset_index(drop=True)
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(model_movies["tags"])
    similarity = cosine_similarity(vectors).astype("float32")
    print(f"Similarity matrix size: {similarity.nbytes / (1024**2):.2f} MB")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / "movie_list.pkl").open("wb") as file:
        pickle.dump(model_movies, file)
    with (MODEL_DIR / "similarity.pkl").open("wb") as file:
        pickle.dump(similarity, file)

    print(f"Saved {len(model_movies)} movies.")
    print(f"Movie list: {MODEL_DIR / 'movie_list.pkl'}")
    print(f"Similarity matrix: {MODEL_DIR / 'similarity.pkl'}")
    print("Next run: streamlit run app.py")


if __name__ == "__main__":
    main()
