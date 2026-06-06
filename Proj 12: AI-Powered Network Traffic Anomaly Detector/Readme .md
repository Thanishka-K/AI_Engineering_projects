# Project 12: Network Traffic Anomaly Detector (API)

## Overview
A moderate-level infrastructure defense microservice that leverages an unsupervised Isolation Forest model to intercept, analyze, and isolate irregular network telemetry metrics via asynchronous API endpoints.

## Technical Details
- **Core Engine Algorithm**: Isolation Forest (`IsolationForest`)
- **Serving Architecture**: FastAPI REST Routing Endpoint
- **Payload Protocol**: Pydantic Strict Data Models

## How to Run
1. Boot up the network telemetry handler instance:
   ```bash
   python app.py
   
