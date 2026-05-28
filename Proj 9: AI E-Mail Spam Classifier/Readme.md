# Project 9: AI E-Mail Spam Classifier

## Overview
A moderate-level Natural Language Processing (NLP) utility that trains a Naive Bayes classifier entirely locally to distinguish between spam and legitimate messages.

## Technical Details
- **Algorithm**: Multinomial Naive Bayes (`MultinomialNB`)
- **Feature Extraction**: Bag-of-Words Text Tokenization via `CountVectorizer`
- **Pipeline Architecture**: Scikit-Learn `Pipeline` for seamless vectorization and inference coupling.

## How to Run
1. Install the required dependencies:
```bash
   pip install -r requirements.txt

