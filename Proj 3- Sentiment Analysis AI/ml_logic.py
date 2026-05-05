from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Simple training data to "warm up" the model
# In a real app, we load a saved .pkl model here
def get_trained_pipeline():
    X = ["I love this", "This is great", "I hate this", "This is bad", "Amazing work", "Very poor"]
    y = ["Positive", "Positive", "Negative", "Negative", "Positive", "Negative"]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    
    pipeline.fit(X, y)
    return pipeline

model = get_trained_pipeline()

def predict_sentiment(text: str):
    prediction = model.predict([text])
    return prediction[0]
