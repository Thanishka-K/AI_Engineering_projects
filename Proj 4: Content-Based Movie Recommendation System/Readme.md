# Project 4: Content-Based Movie Recommender

## Overview
A Machine Learning system that recommends movies based on textual tags (genres, keywords, cast).

## How it Works
1. **Vectorization**: Uses `CountVectorizer` to convert text tags into numerical vectors.
2. **Similarity Score**: Uses **Cosine Similarity** to calculate the distance between movie vectors.
3. **Frontend**: Built with **Streamlit** for a simple user interface.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python model_training.py`
3. Launch the app: `streamlit run app.py`

