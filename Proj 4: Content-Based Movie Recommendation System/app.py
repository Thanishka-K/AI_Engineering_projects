import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie_name):
    index = movies[movies['title'] == movie_name].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = []
    for i in distances[1:4]:  # Get top 3 recommendations
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# UI Layout
st.title('Movie Recommender System')

selected_movie = st.selectbox(
    'Which movie did you like?',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)
