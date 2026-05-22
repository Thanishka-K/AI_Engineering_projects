# Project 8: Multimodal Expense Tracker & Vision Analyzer

## Overview
A high-performance document parsing and ledger analysis pipeline using Multimodal LLMs to automatically ingest images of invoices/receipts, run structural zero-shot visual entity recognition, and yield schema-validated financial databases.

## Technical Architecture
- **Inference Layer**: Gemini 2.5 Flash API
- **Data Validation Layer**: Pydantic Structured Data Wrappers
- **Analytical Portal Layer**: Streamlit Engine & Pandas Frame Components

## Execution Map
1. **Initialize Dependencies**:
   ```bash
   pip install -r requirements.txt
2. streamlit run app.py
