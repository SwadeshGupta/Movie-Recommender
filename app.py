import streamlit as st
from recommender import load_model, recommend_movies, fetch_poster

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommender System")
st.write("Choose a movie and discover films with similar content.")

@st.cache_resource
def get_model():
    return load_model()

try:
    movies, similarity = get_model()
except FileNotFoundError:
    st.error("Model files are missing. Follow the README instructions to add the datasets and run build_model.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the recommendation model: {exc}")
    st.stop()

if movies.empty:
    st.error("The saved movie list is empty.")
    st.stop()

movie_titles = movies["title"].astype(str).tolist()
selected_movie = st.selectbox("Select a movie", movie_titles)

if st.button("Recommend movies", type="primary"):
    try:
        results = recommend_movies(selected_movie, movies, similarity, n=5)
        if not results:
            st.warning("No recommendations were found for this movie.")
        else:
            columns = st.columns(len(results))
            for column, item in zip(columns, results):
                with column:
                    poster = fetch_poster(item["movie_id"])
                    if poster:
                        st.image(poster, use_container_width=True)
                    else:
                        st.caption("Poster unavailable")
                    st.markdown(f"**{item['title']}**")
    except Exception as exc:
        st.error(f"Recommendation failed: {exc}")
