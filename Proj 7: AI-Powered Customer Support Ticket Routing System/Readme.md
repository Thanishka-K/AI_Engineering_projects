# Project 7: Automated Support Ticket Router

## Overview
An operational AI microservice designed to ingest customer text complaints, classify the operational domain using an NLP classification model pipeline, and dispatch immediate assignment logs.

## Technical Architecture
- **Classifier Underlying Logic**: Term Frequency-Inverse Document Frequency (TF-IDF) + Multinomial Naive Bayes.
- **Serving layer**: FastAPI with asynchronous endpoints.
- **Payload Validation**: Pydantic structured schemas.

## Execution Matrix
1. **Train & Serialize Pipeline**:
   ```bash
   python model_training.py
   
