import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_model():
    data = {
        'movie_id': [1, 2, 3, 4],
        'title': ['The Dark Knight', 'Batman Begins', 'Toy Story', 'Finding Nemo'],
        'tags': [
            'action batman joker dc comics dark knight',
            'action batman origins dc comics',
            'animation kids pixar toys adventure',
            'animation kids pixar fish ocean'
        ]
    }
    df = pd.DataFrame(data)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors) #measures the angle between vectors to determine similarity

    pickle.dump(df, open('movie_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    print("Model trained and files saved!")

if __name__ == "__main__":
    train_model()
  
