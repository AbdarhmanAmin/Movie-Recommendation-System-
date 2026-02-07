import streamlit as st
import pickle
import pandas as pd
import zipfile
import os


if not os.path.exists("similarity.pkl"):
    with zipfile.ZipFile("similarity.zip", 'r') as zip_ref:
        zip_ref.extractall()


with open("movies.pkl", "rb") as f:
    df = pickle.load(f)

with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)


def recommend(movie_name):
    if movie_name not in df['title'].values:
        return ["Movie not found in database!"]
    
    movie_index = df[df['title'] == movie_name].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]
    
    recommended_movies = [
        df.iloc[i[0]].title
        for i in movie_list
        if df.iloc[i[0]].title != movie_name
    ]
    
    return recommended_movies


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommender App")
st.write("اكتب اسم الفيلم أو اختاره من القائمة، واحنا هنعملك اقتراحات لأفلام مشابهة.")


movie_input = st.text_input("اكتب اسم الفيلم هنا:")
selected_movie = st.selectbox(
    "أو اختار فيلم من القائمة:",
    df['title'].values
)


if movie_input.strip() != "":
    movie_name = movie_input.strip()
else:
    movie_name = selected_movie


if st.button("Show Recommendations"):
    recommendations = recommend(movie_name)
    
    if "not found" in recommendations[0].lower():
        st.error("الفيلم الي كتبتو مش موجود - اتأكد انك كتبتو بشكل صحيح")
    else:
        st.subheader("أفلام مشابهة:")
       
        cols = st.columns(2)
        for idx, movie in enumerate(recommendations):
            with cols[idx % 2]:
                st.markdown(f"🎥 **{movie}**")
