import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build_spam_classifier():
    data = {
        'text': [
            'Hey, are we still meeting for lunch today at 1 PM?',
            'WINNER! You have won a free $1000 Walmart gift card. Call now!',
            'Can you please send me the updated notes for the AI project?',
            'URGENT! Your account has been compromised. Click here to reset.',
            'Let me know when you are free to discuss the homework assignment.',
            'FREE entry into our weekly prize draw! Text CLAIM to 81188 to win.'
        ],
        'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam']
    }
    df = pd.DataFrame(data)

    # CountVectorizer converts raw text messages into a matrix of token counts
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])

    print("🤖 Training local Naive Bayes Spam Classifier...")
    pipeline.fit(df['text'], df['label'])
    print("✅ Model training complete.\n")

    print("==================================================")
    print("💬 Test the Classifier with your own messages!")
    print("==================================================")
    
    test_messages = [
        "Urgent: You won a cash prize, click this link immediately!",
        "Hey, did you finish editing the README file for the repo?"
    ]

    for msg in test_messages:
        prediction = pipeline.predict([msg])[0]
        probabilities = pipeline.predict_proba([msg])[0]
        confidence = max(probabilities) * 100
        
        status = "🚨 SPAM" if prediction == "spam" else "✅ LEGITIMATE (HAM)"
        print(f"\nMessage: \"{msg}\"")
        print(f"Analysis: {status} ({confidence:.2f}% confidence)")
    print("==================================================")

if __name__ == "__main__":
    build_spam_classifier()
  
