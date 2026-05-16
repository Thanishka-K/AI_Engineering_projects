import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

def train_routing_model():
    # 1. Dataset representing typical enterprise tickets
    data = {
        'text': [
            'My screen is flickering and the server connection dropped',
            'I was double charged for my monthly premium subscription',
            'How do I reset my account password for the portal?',
            'Where can I submit my tax documents for onboarding?',
            'The API returns a 500 internal server error during checkout',
            'Requesting a refund for an accidental transaction invoice'
        ],
        'department': ['Technical', 'Billing', 'Technical', 'HR', 'Technical', 'Billing']
    }
    df = pd.DataFrame(data)

    # 2. Build an End-to-End Pipeline
    # TfidfVectorizer converts text queries into weighted numerical metrics
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])

    # 3. Fit Model
    print("Training Intelligent Ticket Routing Pipeline...")
    pipeline.fit(df['text'], df['department'])

    # 4. Export Artifacts
    with open('ticket_router.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Pipeline exported successfully as 'ticket_router.pkl'")

if __name__ == "__main__":
    train_routing_model()
  
