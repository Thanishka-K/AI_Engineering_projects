import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
df = pd.read_csv('movies.csv') # Example dataset with 'title' and 'tags'

# 2. Vectorize the tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# 3. Calculate Similarity
similarity = cosine_similarity(vectors)

# 4. Recommendation Function
def recommend(movie):
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(df.iloc[i[0]].title)
